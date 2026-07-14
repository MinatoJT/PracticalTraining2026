import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .qwen_vl_client import QwenVLClient
from .visual_query import build_visual_candidate_view, parse_visual_query_target, token_overlap


class VisualCandidatePipeline:
    """共享视觉入口：Qwen 锚点、候选合并、Qwen 重排和旧逻辑兜底。"""
    ANCHOR_PROMPT_VERSION = "dashscope-anchor-v1"
    RERANK_PROMPT_VERSION = "dashscope-pure-visual-rerank-v2"

    def __init__(self):
        self.enabled = os.getenv("VISION_ENABLED", "1") == "1"
        self.rerank_enabled = os.getenv("VISION_RERANK_ENABLED", "1") == "1"
        self.rerank_mode = os.getenv("VISION_RERANK_MODE", "pure_visual").strip().lower()
        self.client = QwenVLClient.shared() if self.enabled else None
        self.rerank_top_n = int(os.getenv("VISION_RERANK_TOP_N", "10"))
        self.min_confidence = float(os.getenv("VISION_MIN_CONFIDENCE", "0.35"))
        self.anchor_fallback_confidence = float(os.getenv("VISION_ANCHOR_FALLBACK_CONFIDENCE", "0.70"))
        self.cache_enabled = os.getenv("VISION_ANCHOR_CACHE", "1") == "1"
        self.debug_path = os.getenv("VISION_DEBUG_PATH")
        self._anchor_cache: Dict[str, Dict[str, Any]] = {}
        self._rerank_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._pipeline_fallbacks = 0
        self._lock = threading.Lock()

    def prepare(self, query: str, image: Image.Image, history: List[Dict[str, Any]], legacy_candidates: List[Dict[str, Any]], cached_anchor: Optional[Dict[str, Any]] = None, refresh_anchor: bool = True, trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return self._fallback(legacy_candidates, "vision_disabled")
        if not isinstance(image, Image.Image):
            return self._fallback(legacy_candidates, "image_encode_error")
        image_hash, history_text = self._image_hash(image), self._history_text(history)
        anchor, analyze_latency = dict(cached_anchor or {}), 0.0
        request_metadata: List[Dict[str, Any]] = []
        if refresh_anchor or not anchor:
            context_fingerprint = hashlib.sha256(
                (query.strip().lower() + "\n" + history_text).encode("utf-8")
            ).hexdigest()[:20]
            cache_key = f"{image_hash}:{context_fingerprint}:{getattr(self.client, 'anchor_model', '')}:{self.ANCHOR_PROMPT_VERSION}"
            with self._lock:
                anchor_result = self._anchor_cache.get(cache_key) if self.cache_enabled else None
                if anchor_result:
                    self._cache_hits += 1
            if not anchor_result:
                anchor_result = self.client.analyze_image(image, query, history_text)
                if anchor_result.get("ok") and self.cache_enabled:
                    with self._lock: self._anchor_cache[cache_key] = anchor_result
            analyze_latency = float(anchor_result.get("latency", 0.0) or 0.0)
            request_metadata.append(self._request_metadata(anchor_result))
            if not anchor_result.get("ok"):
                result = self._fallback(legacy_candidates, anchor_result.get("error", "anchor_failed"))
                result["request_metadata"] = request_metadata
                self._log(trace, query, image_hash, {}, [], result, analyze_latency, 0.0)
                return result
            anchor = anchor_result["anchor"]
        merged = self._merge_candidates(legacy_candidates, anchor)
        limited = merged[:max(1, min(15, self.rerank_top_n))]
        if not limited:
            return self._fallback(legacy_candidates, "rerank_no_candidate", anchor)

        if not self.rerank_enabled:
            ranked = []
            for item in limited:
                candidate = dict(item)
                candidate["qwen_final_score"] = float(candidate.get("anchor_confidence", 0.0) or 0.0)
                candidate["rule_score"] = candidate["qwen_final_score"]
                ranked.append(candidate)
            ranked.sort(key=lambda item: (item.get("qwen_final_score", 0.0), item.get("score", 0.0)), reverse=True)
            selected = ranked[0] if ranked else None
            if selected:
                selected["visual_anchor"] = anchor
                selected["qwen_selected_confidence"] = anchor.get("confidence", 0.0)
            result = {"anchor": anchor, "candidates": ranked, "selected_entity": selected, "rerank": {}, "fallback_used": False, "fallback_reason": "", "fallback_level": "none", "selection_source": "qwen_anchor", "target": parse_visual_query_target(query, anchor), "mode": "anchor_only", "request_metadata": request_metadata}
            self._log(trace, query, image_hash, anchor, merged, result, analyze_latency, 0.0)
            return result

        target = parse_visual_query_target(query, anchor)
        candidate_fingerprint = hashlib.sha256(
            json.dumps(
                [build_visual_candidate_view(item) for item in limited],
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
        target_fingerprint = hashlib.sha256(json.dumps(target, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        rerank_key = f"{image_hash}:{target_fingerprint}:{candidate_fingerprint}:{getattr(self.client, 'rerank_model', '')}:{self.rerank_mode}:{self.RERANK_PROMPT_VERSION}"
        with self._lock:
            rerank_result = self._rerank_cache.get(rerank_key) if self.cache_enabled else None
            if rerank_result:
                self._cache_hits += 1
        if not rerank_result:
            rerank_result = self.client.rerank_candidates(image, query, anchor, limited, history_text)
            if rerank_result.get("ok") and self.cache_enabled:
                with self._lock:
                    self._rerank_cache[rerank_key] = rerank_result
        rerank_latency = float(rerank_result.get("latency", 0.0) or 0.0)
        request_metadata.append(self._request_metadata(rerank_result))
        if not rerank_result.get("ok"):
            result = self._hierarchical_fallback(
                legacy_candidates, merged, anchor,
                rerank_result.get("error", "rerank_failed"),
                rerank_result.get("rerank_diagnostics", {}),
            )
            result["rerank_raw"] = str(rerank_result.get("raw", ""))[:800]
            result["request_metadata"] = request_metadata
            self._log(trace, query, image_hash, anchor, merged, result, analyze_latency, rerank_latency)
            return result
        rerank = rerank_result["rerank"]
        if rerank.get("no_valid_candidate") or float(rerank.get("confidence", 0.0) or 0.0) < self.min_confidence:
            reason = "rerank_no_candidate" if rerank.get("no_valid_candidate") else "low_vision_confidence"
            result = self._hierarchical_fallback(legacy_candidates, merged, anchor, reason, rerank_result.get("rerank_diagnostics", {}))
            result["rerank_raw"] = str(rerank_result.get("raw", ""))[:800]
            result["request_metadata"] = request_metadata
            self._log(trace, query, image_hash, anchor, merged, result, analyze_latency, rerank_latency)
            return result
        score_map = {item["index"]: item for item in rerank.get("candidate_scores", [])}
        ranked = []
        for index, candidate in enumerate(limited, 1):
            item, score = dict(candidate), score_map.get(index, {})
            final_score = score.get("visual_final_score", score.get("final_score"))
            final_score = float(final_score) if final_score is not None else 0.0
            item.update({
                "qwen_appearance_score": score.get("appearance_match", score.get("visual_match")),
                "qwen_target_reference_score": score.get("target_reference_match", score.get("question_match")),
                "qwen_category_score": score.get("category_match"),
                "qwen_ocr_score": score.get("ocr_match"),
                "qwen_depiction_score": score.get("depiction_level_match"),
                "qwen_scene_score": score.get("scene_consistency"),
                "qwen_reason": score.get("reason", ""),
                "qwen_final_score": final_score,
                "rule_score": final_score,
            })
            ranked.append(item)
        ranked.sort(key=lambda item: (item.get("qwen_final_score", 0.0), item.get("score", 0.0)), reverse=True)
        selected_index = int(rerank.get("selected_index", 0) or 0)
        original = limited[selected_index - 1] if 1 <= selected_index <= len(limited) else ranked[0]
        selected = next((item for item in ranked if item.get("entity_name") == original.get("entity_name")), ranked[0])
        selected["visual_anchor"] = anchor
        selected["qwen_selected_confidence"] = rerank.get("confidence", 0.0)
        result = {
            "anchor": anchor, "candidates": ranked, "selected_entity": selected,
            "rerank": rerank, "fallback_used": False, "fallback_reason": "",
            "fallback_level": "none", "selection_source": "qwen_rerank",
            "rerank_diagnostics": rerank_result.get("rerank_diagnostics", {}),
            "rerank_raw": str(rerank_result.get("raw", ""))[:800],
            "target": target, "mode": "anchor_rerank", "request_metadata": request_metadata,
        }
        self._log(trace, query, image_hash, anchor, merged, result, analyze_latency, rerank_latency)
        return result

    def _merge_candidates(self, legacy: List[Dict[str, Any]], anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
        merged = {}
        for raw in legacy:
            item = dict(raw); item["source"] = item.get("source", "image_kg"); item["sources"] = list(dict.fromkeys(item.get("sources", []) + [item["source"]]))
            key = self._normalize_name(item.get("entity_name"));
            if key: merged[key] = item
        anchor_items = list(anchor.get("candidate_entities", [])); primary = str(anchor.get("primary_subject", "")).strip()
        if primary: anchor_items.insert(0, {"name": primary, "confidence": anchor.get("confidence", 0.0), "visual_reason": anchor.get("scene_summary", "")})
        for raw in anchor_items:
            name, key = str(raw.get("name", "")).strip(), self._normalize_name(raw.get("name"))
            if not key: continue
            if key in merged:
                merged[key]["sources"] = list(dict.fromkeys(merged[key].get("sources", []) + ["qwen_anchor"])); merged[key]["anchor_confidence"] = raw.get("confidence", 0.0)
            else:
                merged[key] = {"entity_name": name, "attributes": {"visual_reason": raw.get("visual_reason", "")}, "score": 0.0, "rule_score": 0.0, "source_url": "", "source": "qwen_anchor", "sources": ["qwen_anchor"], "anchor_confidence": raw.get("confidence", 0.0)}
        # 确保视觉锚点候选进入重排窗口，其余仍按 Image-KG 原召回顺序保留。
        return sorted(
            merged.values(),
            key=lambda item: ("qwen_anchor" in item.get("sources", []), item.get("anchor_confidence", 0.0)),
            reverse=True,
        )

    def _fallback(self, candidates, reason, anchor=None):
        with self._lock:
            self._pipeline_fallbacks += 1
        return {"anchor": anchor or {}, "candidates": candidates, "selected_entity": candidates[0] if candidates else None, "rerank": {}, "fallback_used": True, "fallback_reason": reason or "legacy_fallback"}

    def _hierarchical_fallback(self, legacy, merged, anchor, reason, rerank_diagnostics=None):
        """Keep a strong visual anchor before falling back to the legacy image top1."""
        with self._lock:
            self._pipeline_fallbacks += 1
        anchor_match, method = self._match_anchor_candidate(anchor, merged)
        level = method or "legacy_image_top1"
        selected = anchor_match
        ranked = list(merged or legacy)
        if selected is None:
            category = str((anchor or {}).get("generic_category", "")).strip()
            category_matches = [item for item in ranked if self._candidate_matches_category(item, category)]
            if category_matches:
                category_matches.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
                selected = category_matches[0]
                level = "anchor_category_filter"
        if selected is None:
            selected = legacy[0] if legacy else (ranked[0] if ranked else None)
        if selected is not None:
            selected = dict(selected)
            selected["visual_anchor"] = anchor or {}
            selected["qwen_selected_confidence"] = float((anchor or {}).get("confidence", 0.0) or 0.0)
            ranked = [selected] + [item for item in ranked if self._normalize_name(item.get("entity_name")) != self._normalize_name(selected.get("entity_name"))]
        return {
            "anchor": anchor or {}, "candidates": ranked, "selected_entity": selected,
            "rerank": {}, "fallback_used": True, "fallback_reason": reason or "rerank_failed",
            "fallback_level": level, "selection_source": level,
            "anchor_match_candidate": selected.get("entity_name") if selected and level.startswith("anchor_") else "",
            "anchor_match_method": method, "rerank_diagnostics": rerank_diagnostics or {},
            "mode": "anchor_fallback" if level != "legacy_image_top1" else "legacy",
        }

    def _match_anchor_candidate(self, anchor, candidates):
        confidence = float((anchor or {}).get("confidence", 0.0) or 0.0)
        if confidence < self.anchor_fallback_confidence:
            return None, ""
        names = []
        primary = str((anchor or {}).get("primary_subject", "")).strip()
        if primary:
            names.append((primary, confidence))
        for item in sorted((anchor or {}).get("candidate_entities", []) or [], key=lambda value: float(value.get("confidence", 0.0) or 0.0), reverse=True):
            names.append((str(item.get("name", "")).strip(), float(item.get("confidence", 0.0) or 0.0)))
        for name, item_confidence in names:
            if not name or item_confidence < self.anchor_fallback_confidence:
                continue
            normalized = self._normalize_name(name)
            for candidate in candidates:
                candidate_name = self._normalize_name(candidate.get("entity_name"))
                if normalized == candidate_name:
                    return candidate, "anchor_exact_match"
                aliases = build_visual_candidate_view(candidate).get("visual_attributes", {}).get("aliases", "")
                if normalized and normalized in {self._normalize_name(alias) for alias in re.split(r"[,;/|]", str(aliases)) if alias.strip()}:
                    return candidate, "anchor_alias_match"
            best = max(candidates, key=lambda item: token_overlap(name, item.get("entity_name")), default=None)
            if best is not None and token_overlap(name, best.get("entity_name")) >= 0.75:
                return best, "anchor_token_match"
        return None, ""

    @staticmethod
    def _candidate_matches_category(candidate, category):
        if not category:
            return False
        view = build_visual_candidate_view(candidate)
        searchable = " ".join([view.get("entity_name", ""), *view.get("visual_attributes", {}).values()])
        return token_overlap(category, searchable) >= 0.5
    @staticmethod
    def _normalize_name(value): return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower().replace("-", " ")))
    @staticmethod
    def _request_metadata(result):
        return {
            "request_type": result.get("request_type", ""),
            "requested_model": result.get("requested_model", ""),
            "response_model": result.get("response_model", ""),
            "fallback_model_used": bool(result.get("fallback_model_used")),
            "fallback_reason": result.get("fallback_reason", ""),
            "model_mismatch": bool(result.get("model_mismatch")),
            "error": result.get("error", ""),
            "schema_error": result.get("schema_error", ""),
        }
    @staticmethod
    def _history_text(history):
        # 视觉目标只依赖用户问题历史，避免把可能错误的旧 assistant 回答写入锚点缓存键。
        users = [x for x in (history or []) if x.get("role") == "user"][-6:]
        return "\n".join(f"user: {str(x.get('content', ''))[:300]}" for x in users)
    @staticmethod
    def _image_hash(image):
        prepared = image.convert("RGB"); return hashlib.sha256(prepared.tobytes() + str(prepared.size).encode("ascii")).hexdigest()

    def _log(self, trace, query, image_hash, anchor, merged, result, analyze_latency, rerank_latency):
        if not self.debug_path: return
        payload = {"event": "vision_pipeline", "conversation": (trace or {}).get("session_id"), "turn": (trace or {}).get("turn_idx"), "query": query, "target": result.get("target", parse_visual_query_target(query, anchor)), "image_hash": image_hash, "anchor_model": getattr(self.client, "anchor_model", ""), "rerank_model": getattr(self.client, "rerank_model", ""), "mode": result.get("mode", "legacy"), "requests": result.get("request_metadata", []), "anchor_success": bool(anchor), "anchor": anchor, "anchor_confidence": anchor.get("confidence", 0.0) if anchor else 0.0, "visible_text": anchor.get("visible_text", []) if anchor else [], "retrieval_queries": anchor.get("retrieval_queries", []) if anchor else [], "legacy_candidates": [{"entity": x.get("entity_name"), "image_score": x.get("score"), "rule_score": x.get("rule_score")} for x in merged[:15]], "candidate_sources": [{"entity": x.get("entity_name"), "sources": x.get("sources", [])} for x in merged[:15]], "visual_candidate_views": [build_visual_candidate_view(x) for x in merged[:15]], "rerank_raw_response": result.get("rerank_raw", ""), "rerank_parsed_response": result.get("rerank", {}), "rerank_parse_status": (result.get("rerank_diagnostics") or {}).get("parse_status", "not_available"), "recovered_fields": (result.get("rerank_diagnostics") or {}).get("recovered_fields", []), "anchor_match_candidate": result.get("anchor_match_candidate", ""), "anchor_match_method": result.get("anchor_match_method", ""), "fallback_level": result.get("fallback_level", "none"), "selected_entity": (result.get("selected_entity") or {}).get("entity_name"), "selection_source": result.get("selection_source", "legacy_image_top1"), "selected_confidence": (result.get("rerank") or {}).get("confidence", 0.0), "fallback_used": result.get("fallback_used", True), "fallback_reason": result.get("fallback_reason", ""), "cache_hits_total": self._cache_hits, "analyze_latency": round(analyze_latency, 4), "rerank_latency": round(rerank_latency, 4), "usage": self.client.stats() if self.client and hasattr(self.client, "stats") else {}, "final_candidates": [x.get("entity_name") for x in result.get("candidates", [])[:15]]}
        try:
            path = Path(self.debug_path); path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception: pass

    def stats(self) -> Dict[str, Any]:
        """返回不含 Key 和图片内容的视觉调用统计。"""
        client_stats = self.client.stats() if self.client and hasattr(self.client, "stats") else {}
        return {
            "enabled": self.enabled,
            "rerank_enabled": self.rerank_enabled,
            "rerank_mode": self.rerank_mode,
            "cache_hits": self._cache_hits,
            "pipeline_fallbacks": self._pipeline_fallbacks,
            **client_stats,
        }
