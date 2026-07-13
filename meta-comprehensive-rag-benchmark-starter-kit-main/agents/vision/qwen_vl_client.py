import base64
import hashlib
import io
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .visual_anchor import extract_json_object, validate_anchor, validate_rerank


DEFAULT_ANCHOR_MODEL = "qwen3.5-omni-plus"
DEFAULT_RERANK_MODEL = "qwen3.5-omni-flash"
DEFAULT_FALLBACK_MODEL = "qwen3-vl-flash"


def get_model_capabilities(model_name: str) -> Dict[str, bool]:
    """返回当前视觉调用关心的模型能力，禁止 realtime 误走普通 Chat Completions。"""
    normalized = str(model_name or "").strip().lower()
    return {
        "realtime": "-realtime" in normalized,
        "json_mode": normalized in {
            "qwen3.5-omni-plus",
            "qwen3.5-omni-flash",
            "qwen3-vl-flash",
            "qwen3-omni-flash",
        },
        "enable_thinking": normalized == "qwen3-vl-flash",
    }


def resolve_model_config() -> Tuple[str, str, str]:
    """集中解析新旧模型变量；显式的新变量优先于旧 QWEN_VL_MODEL。"""
    legacy = os.getenv("QWEN_VL_MODEL", "").strip()
    anchor = os.getenv("QWEN_VL_ANCHOR_MODEL", legacy or DEFAULT_ANCHOR_MODEL).strip()
    rerank = os.getenv("QWEN_VL_RERANK_MODEL", legacy or DEFAULT_RERANK_MODEL).strip()
    fallback = os.getenv("QWEN_VL_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip()
    return anchor or DEFAULT_ANCHOR_MODEL, rerank or DEFAULT_RERANK_MODEL, fallback or DEFAULT_FALLBACK_MODEL


class QwenVLClient:
    """共享的百炼 Qwen3-VL OpenAI-compatible 客户端。"""

    _instances: Dict[Tuple[str, ...], "QwenVLClient"] = {}
    _lock = threading.Lock()

    @classmethod
    def shared(cls) -> "QwenVLClient":
        provider = os.getenv("QWEN_VL_PROVIDER", "dashscope").strip().lower()
        anchor_model, rerank_model, fallback_model = resolve_model_config()
        base_url = os.getenv(
            "QWEN_VL_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip().rstrip("/")
        raw_key = os.getenv("QWEN_VL_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
        api_key = raw_key.strip().strip('"').strip("'")
        key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12] if api_key else "missing"
        cache_key = (provider, anchor_model, rerank_model, fallback_model, base_url, key_fingerprint)
        with cls._lock:
            if cache_key not in cls._instances:
                cls._instances[cache_key] = cls(
                    anchor_model,
                    base_url,
                    api_key,
                    provider,
                    rerank_model=rerank_model,
                    fallback_model=fallback_model,
                )
            return cls._instances[cache_key]

    def __init__(self, model: str, base_url: str, api_key: str, provider: str = "dashscope", rerank_model: Optional[str] = None, fallback_model: Optional[str] = None):
        self.anchor_model = str(model or DEFAULT_ANCHOR_MODEL).strip()
        self.rerank_model = str(rerank_model or model or DEFAULT_RERANK_MODEL).strip()
        self.fallback_model = str(fallback_model or DEFAULT_FALLBACK_MODEL).strip()
        self.model = self.anchor_model
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = str(api_key or "").strip().strip('"').strip("'")
        self.provider = provider
        self.timeout = float(os.getenv("QWEN_VL_TIMEOUT", "90"))
        self.max_retries = max(0, min(2, int(os.getenv("QWEN_VL_MAX_RETRIES", "2"))))
        self.max_tokens = int(os.getenv("QWEN_VL_MAX_TOKENS", "1024"))
        self.enable_thinking = os.getenv("QWEN_VL_ENABLE_THINKING", "0") == "1"
        self.max_image_edge = int(os.getenv("QWEN_VL_MAX_IMAGE_EDGE", "1280"))
        self.jpeg_quality = max(85, min(100, int(os.getenv("QWEN_VL_JPEG_QUALITY", "92"))))
        self._stats_lock = threading.Lock()
        self._stats = {
            "anchor_calls": 0,
            "rerank_calls": 0,
            "fallbacks": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "latency_total": 0.0,
        }
        self.client = self._build_client()

    def _build_client(self):
        if not self.api_key:
            return None
        if not self.base_url.startswith(("https://", "http://")):
            return None
        try:
            from openai import OpenAI

            # 重试由本类控制，避免 SDK 和外层叠加后超过两次。
            return OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        except Exception:
            return None

    def health_check(self, model: Optional[str] = None, allow_fallback: bool = False) -> Dict[str, Any]:
        """使用最小文本 Chat Completion 检查服务，不依赖 /v1/models。"""
        result = self._chat_json(
            messages=[{"role": "user", "content": "Return JSON exactly as {\"ok\": true}."}],
            purpose="health",
            max_tokens=32,
            model=model or self.anchor_model,
            allow_model_fallback=allow_fallback,
        )
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error", "health_check_failed")}
        parsed, error = extract_json_object(result.get("content", ""))
        return {
            "ok": not error and bool(parsed.get("ok")),
            "error": error,
            "model": result.get("response_model", ""),
            "requested_model": result.get("requested_model", ""),
            "response_model": result.get("response_model", ""),
            "original_requested_model": result.get("original_requested_model", ""),
            "fallback_model_used": bool(result.get("fallback_model_used")),
            "fallback_reason": result.get("fallback_reason", ""),
            "finish_reason": result.get("finish_reason", ""),
            "usage": result.get("usage", {}),
        }

    def analyze_image(self, image: Image.Image, query: str, history: str = "") -> Dict[str, Any]:
        result = self._vision_json(image, self._anchor_prompt(query, history), "anchor", self.anchor_model)
        if not result.get("ok"):
            return result
        parsed, error = extract_json_object(result.get("content", ""))
        if error:
            return {**result, "ok": False, "error": "invalid_json", "raw": result.get("content", "")[:800]}
        anchor, error = validate_anchor(parsed, max_candidates=5, max_queries=5)
        if error:
            return {**result, "ok": False, "error": "schema_validation_failed", "schema_error": error, "raw": result.get("content", "")[:800]}
        return {**result, "anchor": anchor, "raw": result.get("content", "")[:800]}

    def rerank_candidates(
        self,
        image: Image.Image,
        query: str,
        anchor: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        history: str = "",
    ) -> Dict[str, Any]:
        result = self._vision_json(image, self._rerank_prompt(query, anchor, candidates, history), "rerank", self.rerank_model)
        if not result.get("ok"):
            return result
        parsed, error = extract_json_object(result.get("content", ""))
        if error:
            return {**result, "ok": False, "error": "invalid_json", "raw": result.get("content", "")[:800]}
        rerank, error = validate_rerank(parsed, len(candidates))
        if error:
            mapped = "selected_index_out_of_range" if "index" in error else "schema_validation_failed"
            return {**result, "ok": False, "error": mapped, "schema_error": error, "raw": result.get("content", "")[:800]}
        selected_index = int(rerank.get("selected_index", 0) or 0)
        if not rerank.get("no_valid_candidate"):
            expected = str(candidates[selected_index - 1].get("entity_name", "")).strip().lower()
            returned = str(rerank.get("selected_entity", "")).strip().lower()
            if expected != returned:
                return {**result, "ok": False, "error": "schema_validation_failed", "schema_error": "selected_entity_mismatch"}
        if not rerank.get("no_valid_candidate") and not rerank.get("candidate_scores"):
            return {**result, "ok": False, "error": "schema_validation_failed", "schema_error": "candidate_scores_missing"}
        return {**result, "rerank": rerank, "raw": result.get("content", "")[:800]}

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            values = dict(self._stats)
        calls = values["anchor_calls"] + values["rerank_calls"]
        values["average_latency"] = round(values["latency_total"] / calls, 4) if calls else 0.0
        values["average_tokens"] = round(values["total_tokens"] / calls, 2) if calls else 0.0
        return values

    def _vision_json(self, image: Image.Image, prompt: str, purpose: str, model: str) -> Dict[str, Any]:
        try:
            data_url = self._image_data_url(image)
        except Exception:
            return {"ok": False, "error": "image_encode_failed", "latency": 0.0, "retry_count": 0}
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }]
        return self._chat_json(messages, purpose, self.max_tokens, model=model, allow_model_fallback=True)

    def _chat_json(
        self,
        messages: List[Dict[str, Any]],
        purpose: str,
        max_tokens: int,
        model: Optional[str] = None,
        allow_model_fallback: bool = True,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "error": "missing_api_key", "latency": 0.0, "retry_count": 0}
        if not self.base_url.startswith(("https://", "http://")):
            return {"ok": False, "error": "invalid_base_url", "latency": 0.0, "retry_count": 0}
        if self.client is None:
            return {"ok": False, "error": "client_unavailable", "latency": 0.0, "retry_count": 0}

        requested_model = str(model or self.anchor_model).strip()
        if get_model_capabilities(requested_model)["realtime"]:
            return self._failure("realtime_model_not_supported", 0.0, 0, purpose, requested_model)

        result = self._request_model(messages, purpose, max_tokens, requested_model)
        fallback_errors = {
            "model_not_found",
            "model_not_available",
            "unsupported_parameter",
            "temporary_server_error",
        }
        if (
            allow_model_fallback
            and not result.get("ok")
            and result.get("error") in fallback_errors
            and self.fallback_model
            and self.fallback_model != requested_model
        ):
            if get_model_capabilities(self.fallback_model)["realtime"]:
                return result
            fallback = self._request_model(messages, purpose, max_tokens, self.fallback_model)
            fallback["fallback_model_used"] = True
            fallback["fallback_reason"] = result.get("error", "")
            fallback["original_requested_model"] = requested_model
            return fallback
        return result

    def _request_model(self, messages: List[Dict[str, Any]], purpose: str, max_tokens: int, model: str) -> Dict[str, Any]:
        started = time.perf_counter()
        last_error = "temporary_server_error"
        for attempt in range(self.max_retries + 1):
            try:
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                }
                if get_model_capabilities(model)["enable_thinking"]:
                    request_kwargs["extra_body"] = {"enable_thinking": False}
                response = self.client.chat.completions.create(**request_kwargs)
                latency = time.perf_counter() - started
                choices = list(getattr(response, "choices", None) or [])
                if not choices:
                    return self._failure("choices_empty", latency, attempt, purpose, model)
                choice = choices[0]
                finish_reason = str(getattr(choice, "finish_reason", "") or "")
                content = str(getattr(choice.message, "content", None) or "").strip()
                if not content:
                    return self._failure("content_empty", latency, attempt, purpose, model)
                if finish_reason == "length":
                    return self._failure("content_truncated", latency, attempt, purpose, model)
                usage = self._usage_dict(getattr(response, "usage", None))
                self._record_success(purpose, latency, usage)
                return {
                    "ok": True,
                    "error": "",
                    "content": content,
                    "latency": latency,
                    "retry_count": attempt,
                    "response_model": str(getattr(response, "model", "") or ""),
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "request_type": purpose,
                    "requested_model": model,
                    "fallback_model_used": False,
                    "fallback_reason": "",
                    "model_mismatch": bool(getattr(response, "model", "") and str(getattr(response, "model", "")) != model),
                }
            except Exception as exc:
                last_error, retryable = self._classify_error(exc)
                if not retryable or attempt >= self.max_retries:
                    return self._failure(last_error, time.perf_counter() - started, attempt, purpose, model)
                time.sleep(1 if attempt == 0 else 2)
        return self._failure(last_error, time.perf_counter() - started, self.max_retries, purpose, model)

    def _failure(self, error: str, latency: float, retries: int, purpose: str, model: str = "") -> Dict[str, Any]:
        if purpose in {"anchor", "rerank"}:
            with self._stats_lock:
                self._stats["fallbacks"] += 1
        return {
            "ok": False,
            "error": error,
            "latency": latency,
            "retry_count": retries,
            "request_type": purpose,
            "requested_model": model,
            "response_model": "",
            "fallback_model_used": False,
            "fallback_reason": "",
        }

    def _record_success(self, purpose: str, latency: float, usage: Dict[str, int]) -> None:
        with self._stats_lock:
            if purpose == "anchor":
                self._stats["anchor_calls"] += 1
            elif purpose == "rerank":
                self._stats["rerank_calls"] += 1
            self._stats["prompt_tokens"] += usage["prompt_tokens"]
            self._stats["completion_tokens"] += usage["completion_tokens"]
            self._stats["total_tokens"] += usage["total_tokens"]
            self._stats["cached_tokens"] += usage["cached_tokens"]
            self._stats["latency_total"] += latency

    @staticmethod
    def _usage_dict(usage: Any) -> Dict[str, int]:
        details = getattr(usage, "prompt_tokens_details", None)
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0),
        }

    @staticmethod
    def _classify_error(exc: Exception) -> Tuple[str, bool]:
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__.lower()
        text = repr(exc).lower()
        if "region" in text or "workspace" in text and "mismatch" in text:
            return "workspace_region_mismatch", False
        if status == 401 or "authentication" in name:
            return "authentication_error", False
        if status == 403 or "permission" in text:
            return "permission_denied", False
        if status == 429 or "rate limit" in text:
            return "rate_limited", True
        if status in {500, 502, 503, 504}:
            return "temporary_server_error", True
        if status == 404 and "model" in text:
            return "model_not_found", False
        if status == 404:
            return "model_not_available", False
        if status == 400 and ("unsupported" in text or "unknown parameter" in text):
            return "unsupported_parameter", False
        if status == 400 and "model" in text:
            return "model_not_available", False
        if status == 400:
            return "bad_request", False
        if "timeout" in name or "timeout" in text:
            return "request_timeout", True
        if "connection" in name or "connection" in text:
            return "connection_error", True
        return "temporary_server_error", False

    def _image_data_url(self, image: Image.Image) -> str:
        prepared = image.convert("RGB").copy()
        # thumbnail 保持比例，且不会放大小图。
        prepared.thumbnail((self.max_image_edge, self.max_image_edge))
        buffer = io.BytesIO()
        prepared.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _anchor_prompt(query: str, history: str) -> str:
        return (
            "你是视觉证据分析器，不回答最终知识问题。只根据原图定位当前问题指向的对象。"
            "严格区分现实物体、画作/屏幕/海报/包装/图表本身及其中描绘的物体，并提取相关 OCR。"
            "旧 assistant 回答可能错误。图中不确定时返回通用类别，不要强猜具体实体。"
            "只输出 JSON 对象，不要代码块或长篇解释。candidate_entities 最多5个，retrieval_queries 最多5个。\n"
            "image_type、scene_summary、question_target、primary_subject、generic_category 必须是字符串，"
            "尤其 question_target 只能是简短字符串，不得输出对象或数组；confidence 必须是0到1之间的数字。\n"
            f"当前问题：{query}\n必要历史：{history or 'None'}\n"
            "JSON 字段：image_type, scene_summary, question_target, primary_subject, generic_category, "
            "candidate_entities[{name,confidence,visual_reason}], visible_text[{text,location,confidence}], "
            "visual_attributes{color,shape,material,count,location,other}, is_depiction_inside_another_object, "
            "requires_external_knowledge, confidence, retrieval_queries。"
        )

    @staticmethod
    def _rerank_prompt(query: str, anchor: Dict[str, Any], candidates: List[Dict[str, Any]], history: str) -> str:
        compact = []
        for index, item in enumerate(candidates, 1):
            attrs = item.get("attributes", {}) or {}
            compact.append({
                "index": index,
                "entity": item.get("entity_name", ""),
                "attributes": "; ".join(
                    f"{key}: {' '.join(str(value).split())[:180]}"
                    for key, value in list(attrs.items())[:4]
                ),
                "image_similarity_reference": item.get("score", 0.0),
                "sources": item.get("sources", [item.get("source", "image_kg")]),
            })
        return (
            "你是视觉候选验证器，不回答最终问题。根据原图、当前问题、OCR 和视觉锚点重排候选。"
            "候选 index 使用1-based。图片相似度仅供参考，视觉匹配和问题指向是主要依据。"
            "严格区分画作本身与画中物体。只输出 JSON 对象，不要代码块。\n"
            f"问题：{query}\n历史：{history or 'None'}\n"
            f"视觉锚点：{json.dumps(anchor, ensure_ascii=False)}\n"
            f"候选：{json.dumps(compact, ensure_ascii=False)}\n"
            "JSON 字段：selected_index, selected_entity, confidence, "
            "candidate_scores[{index,visual_match,question_match,ocr_match,final_score,reason}], "
            "rejected_indices, no_valid_candidate。selected_entity 必须与 selected_index 对应。"
        )
