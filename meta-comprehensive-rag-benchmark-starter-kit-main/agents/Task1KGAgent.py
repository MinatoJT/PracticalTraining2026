import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline


class Task1KGAgent(BaseAgent):
    """Task1 知识图谱 Agent：图像检索 KG + 实体选择 + DeepSeek 文本生成。"""

    def __init__(
        self,
        search_pipeline: UnifiedSearchPipeline,
        top_k: int = 15,
        rerank_top_n: int = 6,
        min_score: float = 0.0,
        model_name: Optional[str] = None,
    ):
        super().__init__(search_pipeline)
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.min_score = min_score
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.debug_path = os.getenv("TASK1_DEBUG_PATH")
        self.client = self._build_client()
        self._debug({"event": "init", "has_api_key": bool(self.api_key), "model": self.model_name, "has_client": self.client is not None})

    def get_batch_size(self) -> int:
        # API 逐条生成时 batch 太大容易等待过久；4 是实训调试和速度之间的折中。
        return 4

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        responses = []
        for query, image, history in zip(queries, images, message_histories):
            # 1. 调用官方 Task1 图像检索 API，获得相似图像及其 KG 实体。
            raw_results = self._image_search(image)
            evidence = self._build_evidence(raw_results)

            # 2. 先用规则做问题感知重排，再交给 DeepSeek 在候选实体中选择。
            ranked_evidence = self._rank_candidates_by_rules(query, evidence)
            selected = self._select_entity(query, ranked_evidence)
            self._debug({
                "event": "query",
                "query": query,
                "evidence_count": len(evidence),
                "ranked_entities": [item.get("entity_name") for item in ranked_evidence[:5]],
                "selected_entity": selected.get("entity_name") if selected else None,
                "has_client": self.client is not None,
            })

            # 3. 基于选中的实体回答；没有 API key 时用规则兜底，方便本地 smoke test。
            if self.client is not None:
                answer = self._answer_with_llm(query, selected, ranked_evidence, history)
            else:
                answer = self._answer_with_rules(query, ranked_evidence)
            responses.append(self._finalize_answer(answer))
        return responses

    def _build_client(self):
        # OpenAI SDK 兼容 DeepSeek API；未设置 key 时不初始化，避免测试脚本必须联网。
        if not self.api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _image_search(self, image: Image.Image) -> List[Dict[str, Any]]:
        # Task1 的关键接口：传入 PIL Image，返回相似图像及其 KG 实体，而不是文本检索结果。
        if self.search_pipeline is None:
            return []
        try:
            results = self.search_pipeline(image, k=self.top_k)
        except Exception as exc:
            self._debug({"event": "image_search_error", "error": repr(exc)})
            return []
        return results or []

    def _build_evidence(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 将 image search 原始结果压平成候选实体列表，并保留相似度、属性和来源 URL。
        evidence = []
        seen = set()
        for result in results:
            score = float(result.get("score", 0.0) or 0.0)
            if score < self.min_score:
                continue
            for entity in result.get("entities", []) or []:
                name = self._clean_text(entity.get("entity_name", ""))
                attrs = entity.get("entity_attributes", {}) or {}
                cleaned_attrs = self._clean_attributes(attrs)
                if not name and not cleaned_attrs:
                    continue
                key = (name, json.dumps(cleaned_attrs, sort_keys=True, ensure_ascii=True))
                if key in seen:
                    continue
                seen.add(key)
                evidence.append({
                    "score": round(score, 4),
                    "entity_name": name,
                    "attributes": cleaned_attrs,
                    "source_url": result.get("url", ""),
                })
        return evidence

    def _rank_candidates_by_rules(self, query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 规则重排接口：把图像相似度和问题关键词匹配分加权，减少盲信 top-1 的情况。
        ranked = []
        for item in evidence:
            candidate = dict(item)
            candidate["rule_score"] = self._score_candidate_by_rules(query, item)
            ranked.append(candidate)
        return sorted(ranked, key=lambda item: item.get("rule_score", 0.0), reverse=True)

    def _score_candidate_by_rules(self, query: str, candidate: Dict[str, Any]) -> float:
        # 候选实体打分接口：根据问题类型给实体名和属性命中加分。
        query_l = query.lower()
        name_l = str(candidate.get("entity_name", "")).lower()
        attrs_l = " ".join(f"{k} {v}" for k, v in candidate.get("attributes", {}).items()).lower()
        text_l = f"{name_l} {attrs_l}"

        score = float(candidate.get("score", 0.0) or 0.0)
        groups = [
            (["car", "vehicle", "passenger", "towing", "torque", "engine", "awd", "gallon", "mpg"], ["car", "vehicle", "motor", "automobile", "truck", "sedan", "suv", "ford", "toyota", "honda", "nissan", "subaru", "chevrolet", "jeep", "ram", "dodge"]),
            (["food", "origin", "protein", "edible", "fruit", "skin", "bad", "gone bad"], ["food", "fruit", "dish", "banana", "avocado", "fries", "lentil", "chickpea", "dragon fruit", "carne asada"]),
            (["building", "architect", "floor", "built", "remodel", "tower", "cathedral"], ["building", "tower", "cathedral", "church", "architect", "construction", "floor", "height"]),
            (["cat", "male", "female"], ["cat", "breed", "feline", "calico", "shorthair", "himalayan", "nebelung"]),
            (["color", "wavelength"], ["color", "fruit", "green", "red", "yellow", "avocado"]),
            (["safe", "dangerous", "children", "hearing", "decibel"], ["safety", "airpods", "earbuds", "rake", "tool", "tower", "antenna"]),
            (["police", "olympics", "japanese", "nagano"], ["police", "olympics", "vehicle", "nissan", "car"]),
        ]

        for query_terms, entity_terms in groups:
            if any(term in query_l for term in query_terms):
                score += 0.12
                for term in entity_terms:
                    if term in text_l:
                        score += 0.08

        # 问题中直接出现实体词时额外加分，例如 wrx、mustang、banana 等。
        for token in re.findall(r"[a-z0-9]+", query_l):
            if len(token) >= 4 and token in text_l:
                score += 0.05
        return round(score, 4)

    def _select_entity(self, query: str, ranked_evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # 实体选择总接口：先取规则重排后的前 N 个，再让 DeepSeek 输出最可能实体。
        if not ranked_evidence:
            return None
        candidates = ranked_evidence[: self.rerank_top_n]
        if self.client is None:
            return candidates[0]
        selected = self._select_entity_with_llm(query, candidates)
        return selected or candidates[0]

    def _select_entity_with_llm(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # DeepSeek 实体选择接口：只负责从候选中选实体，不负责回答问题。
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._build_entity_selection_messages(query, candidates),
                temperature=0.0,
                max_tokens=24,
            )
            raw = response.choices[0].message.content or ""
            selected = self._parse_entity_selection(raw, candidates)
            self._debug({"event": "entity_select", "query": query, "raw": raw[:300], "selected": selected.get("entity_name") if selected else None})
            return selected
        except Exception as exc:
            self._debug({"event": "entity_select_error", "query": query, "error": repr(exc)})
            return None

    def _build_entity_selection_messages(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # 实体选择 prompt：中文短提示，只要求返回 INDEX，避免 JSON 约束导致空回复。
        rows = []
        for idx, item in enumerate(candidates, start=1):
            attrs = item.get("attributes", {})
            attr_text = "; ".join(f"{k}: {v}" for k, v in list(attrs.items())[:6])
            rows.append(f"{idx}. 实体={item.get('entity_name')} 图像分={item.get('score')} 规则分={item.get('rule_score')} 属性={attr_text}")
        system = (
            "你是多模态问答中的实体选择器。根据问题和候选实体，选择最可能是图片主体、且最适合回答问题的实体。"
            "只能输出一行，格式必须是：INDEX: 数字。不要解释。"
        )
        user = (
            f"问题：{query}\n"
            "候选实体：\n"
            + "\n".join(rows)
            + "\n请只输出 INDEX: 数字，例如 INDEX: 3。"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _parse_entity_selection(self, raw: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # 解析 DeepSeek 的 INDEX 选择；解析失败时返回 None，由规则排序兜底。
        match = re.search(r"(?:INDEX|index|索引|编号)\s*[:：]?\s*(\d+)", raw)
        if not match:
            match = re.search(r"\b(\d+)\b", raw)
        if not match:
            return None
        try:
            index = int(match.group(1))
            if 1 <= index <= len(candidates):
                selected = dict(candidates[index - 1])
                selected["selection_raw"] = raw
                return selected
        except Exception:
            return None
        return None

    def _answer_with_llm(self, query: str, selected: Optional[Dict[str, Any]], candidates: List[Dict[str, Any]], history: List[Dict[str, Any]]) -> str:
        # DeepSeek 回答接口：只围绕选中的实体回答，避免多个候选让模型犹豫。
        if not selected:
            return "I don't know"
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._build_answer_messages(query, selected, candidates, history),
                temperature=0.1,
                max_tokens=75,
            )
            answer = response.choices[0].message.content or ""
            if self._is_entity_echo(answer, selected):
                answer = self._repair_entity_echo(query, selected, candidates, history)
            if not answer.strip():
                answer = self._answer_with_rules(query, candidates)
            self._debug({"event": "llm_success", "query": query, "selected": selected.get("entity_name"), "answer": answer[:200]})
            return answer
        except Exception as exc:
            self._debug({"event": "llm_error", "query": query, "error": repr(exc)})
            return self._answer_with_rules(query, candidates)

    def _build_answer_messages(self, query: str, selected: Dict[str, Any], candidates: List[Dict[str, Any]], history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # 回答 prompt：中文提示，强调回答问题本身，不要只复述实体名。
        selected_attrs = selected.get("attributes", {})
        attr_text = "; ".join(f"{k}: {v}" for k, v in list(selected_attrs.items())[:16])
        alternatives = ", ".join(item.get("entity_name", "") for item in candidates[:5] if item.get("entity_name") and item.get("entity_name") != selected.get("entity_name"))
        history_text = self._format_history(history)
        system = (
            "你是视觉问答助手。已知图片最可能对应一个实体，请基于该实体、结构化属性和常识回答用户问题。"
            "必须回答问题所问的事实、判断、数值、日期、来源或安全建议；不要只输出实体名。"
            "回答要简洁，通常一句话。不要提到检索、证据或知识图谱。"
        )
        user = (
            f"用户问题：{query}\n"
            f"选中实体：{selected.get('entity_name')}\n"
            f"选中实体属性：{attr_text}\n"
            f"其他可能实体：{alternatives}\n"
            f"历史上下文：{history_text}\n"
            "请直接回答问题本身，不要只复述实体名。"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _is_entity_echo(self, answer: str, selected: Dict[str, Any]) -> bool:
        # 判断回答是否只是复述实体名；这种输出通常会被评测判错。
        normalized_answer = re.sub(r"[^a-z0-9]+", " ", str(answer).lower()).strip()
        normalized_entity = re.sub(r"[^a-z0-9]+", " ", str(selected.get("entity_name", "")).lower()).strip()
        if not normalized_answer or not normalized_entity:
            return False
        return normalized_answer == normalized_entity or normalized_answer in normalized_entity or normalized_entity in normalized_answer

    def _repair_entity_echo(self, query: str, selected: Dict[str, Any], candidates: List[Dict[str, Any]], history: List[Dict[str, Any]]) -> str:
        # 如果模型只返回实体名，则二次追问，强制回答问题所需的属性/判断。
        selected_attrs = selected.get("attributes", {})
        attr_text = "; ".join(f"{k}: {v}" for k, v in list(selected_attrs.items())[:16])
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你刚才只输出了实体名。现在必须回答用户问题本身，不能只输出实体名。答案简洁。"},
                    {"role": "user", "content": f"问题：{query}\n实体：{selected.get('entity_name')}\n属性：{attr_text}\n请回答问题所问的事实、判断、数值、日期、来源或安全建议。"},
                ],
                temperature=0.1,
                max_tokens=75,
            )
            repaired = response.choices[0].message.content or ""
            self._debug({"event": "entity_echo_repair", "query": query, "selected": selected.get("entity_name"), "answer": repaired[:200]})
            return repaired
        except Exception as exc:
            self._debug({"event": "entity_echo_repair_error", "query": query, "error": repr(exc)})
            return ""
    def _answer_with_rules(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        # 规则兜底接口：没有 API key 或 API 出错时，尽量从 KG 字段直接抽答案。
        if not evidence:
            return "I don't know"
        query_l = query.lower()
        field_groups = [
            (["architect", "designed", "designer"], ["architect", "designer", "design"]),
            (["when", "year", "date", "opened", "opening"], ["opening", "opened", "completion_date", "start_date", "date"]),
            (["where", "address", "located", "location"], ["address", "location", "coordinates"]),
            (["how many floors", "floor"], ["floor_count", "floors", "top_floor"]),
            (["height", "tall", "roof"], ["roof", "height", "top_floor"]),
            (["who", "owner"], ["owner", "developer", "manufacturer", "author"]),
            (["what", "name", "called"], ["name", "title", "entity_name"]),
        ]
        for query_terms, attr_terms in field_groups:
            if any(term in query_l for term in query_terms):
                value = self._find_attribute(evidence, attr_terms)
                if value:
                    return value
        best_name = evidence[0].get("entity_name")
        return best_name or "I don't know"

    def _find_attribute(self, evidence: List[Dict[str, Any]], attr_terms: List[str]) -> Optional[str]:
        # 属性查找接口：按字段名模糊匹配 KG 属性。
        for item in evidence:
            attrs = item.get("attributes", {})
            for key, value in attrs.items():
                key_l = key.lower()
                if any(term in key_l for term in attr_terms):
                    return value
            if "entity_name" in attr_terms and item.get("entity_name"):
                return item["entity_name"]
        return None

    def _clean_attributes(self, attrs: Dict[str, Any]) -> Dict[str, str]:
        cleaned = {}
        for key, value in attrs.items():
            clean_key = self._clean_text(str(key))
            clean_value = self._clean_text(str(value))
            if clean_key and clean_value:
                cleaned[clean_key] = clean_value
        return cleaned

    def _clean_text(self, text: str) -> str:
        # 文本清洗接口：清理 KG 中常见的 HTML、Wiki 链接、模板标记。
        text = re.sub(r"<br\s*/?>", "; ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", text)
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        text = text.replace("&quot;", '"').replace("&#x27;", "'").replace("&amp;", "&")
        text = re.sub(r"\s+", " ", text).strip(" ;,\n\t")
        return text

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return "None"
        return "\n".join(f"{msg.get('role')}: {msg.get('content')}" for msg in history[-6:])

    def _finalize_answer(self, answer: str) -> str:
        # 输出整理接口：压缩空白，避免空字符串进入评测。
        answer = re.sub(r"\s+", " ", str(answer or "")).strip()
        if not answer:
            return "I don't know"
        return answer

    def _debug(self, payload: Dict[str, Any]) -> None:
        # 调试日志接口：不记录 API key，只记录检索、实体选择和 API 调用状态。
        if not self.debug_path:
            return
        try:
            path = Path(self.debug_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

