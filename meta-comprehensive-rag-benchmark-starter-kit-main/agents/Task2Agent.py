import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from cragmm_search.search import UnifiedSearchPipeline

# Task2 不修改 Task1；直接在 Task1KGAgent 的基础上继承扩展。
try:
    from agents.Task1KGAgent import Task1KGAgent
except ImportError:
    from Task1KGAgent import Task1KGAgent


class Task2Agent(Task1KGAgent):
    """
    Task2 多源增强 Agent。

    在 Task1KGAgent 的基础上新增：
    1. Web 检索；
    2. Web 结果解析；
    3. Web 噪声过滤与排序；
    4. Image-KG 证据 + Web 证据融合；
    5. 面向多源增强任务的 DeepSeek 回答 prompt。

    证据优先级：
    - Image-KG 是强证据，因为它直接来自图像检索和结构化 KG；
    - Web evidence 是辅助证据，主要用于补充背景知识；
    - 如果 Web 和 Image-KG 冲突，默认优先相信 Image-KG；
    - 如果证据不足，回答 I don't know。
    """

    def __init__(
        self,
        search_pipeline: UnifiedSearchPipeline,
        top_k: int = 15,
        rerank_top_n: int = 6,
        min_score: float = 0.0,
        model_name: Optional[str] = None,
        web_top_k: int = 8,
        web_keep_top_n: int = 4,
        min_web_score: float = 0.08,
    ):
        super().__init__(
            search_pipeline=search_pipeline,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            min_score=min_score,
            model_name=model_name,
        )

        self.web_top_k = web_top_k
        self.web_keep_top_n = web_keep_top_n
        self.min_web_score = min_web_score
        self.task2_debug_path = os.getenv("TASK2_DEBUG_PATH") or self.debug_path

        self._debug_task2({
            "event": "task2_init",
            "web_top_k": self.web_top_k,
            "web_keep_top_n": self.web_keep_top_n,
            "min_web_score": self.min_web_score,
            "has_client": self.client is not None,
            "model": self.model_name,
        })

    def get_batch_size(self) -> int:
        # Task2 每个样本会进行 image search + web search + LLM 生成。
        # 先设为 1，避免 DeepSeek API 调用过慢导致 batch 超时。
        return 1

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        responses = []

        for query, image, history in zip(queries, images, message_histories):
            # 1. 复用 Task1：图像检索 KG。
            raw_image_results = self._image_search(image)
            kg_evidence = self._build_evidence(raw_image_results)
            ranked_kg = self._rank_candidates_by_rules(query, kg_evidence)
            selected_entity = self._select_entity(query, ranked_kg)

            # 2. Task2 新增：基于问题和 KG 实体构造网页检索 query。
            web_query = self._build_web_query(query, selected_entity, ranked_kg)
            raw_web_results = self._web_search(web_query)
            web_evidence = self._build_web_evidence(raw_web_results)

            # 3. Task2 新增：网页证据过滤、排序、保留 top N。
            ranked_web = self._rank_web_evidence(
                query=query,
                web_query=web_query,
                web_evidence=web_evidence,
                selected_entity=selected_entity,
                kg_evidence=ranked_kg,
            )

            # 4. Task2 新增：KG-Web 多源融合。
            fused_context = self._fuse_multisource_evidence(
                query=query,
                selected_entity=selected_entity,
                kg_evidence=ranked_kg,
                web_evidence=ranked_web,
            )

            self._debug_task2({
                "event": "task2_query",
                "query": query,
                "selected_entity": selected_entity.get("entity_name") if selected_entity else None,
                "kg_count": len(kg_evidence),
                "web_query": web_query,
                "web_count": len(web_evidence),
                "ranked_web_titles": [item.get("title") for item in ranked_web[: self.web_keep_top_n]],
                "has_client": self.client is not None,
            })

            # 5. 多源回答。
            if self.client is not None:
                answer = self._answer_task2_with_llm(
                    query=query,
                    selected_entity=selected_entity,
                    kg_candidates=ranked_kg,
                    web_evidence=ranked_web,
                    fused_context=fused_context,
                    history=history,
                )
            else:
                # 无 DeepSeek key 或 SDK 不可用时，回退到 Task1 的 KG 规则抽取。
                answer = self._answer_with_rules(query, ranked_kg)

            responses.append(self._finalize_answer(answer))

        return responses

    # ---------------------------------------------------------------------
    # Web Search
    # ---------------------------------------------------------------------

    def _build_web_query(
        self,
        query: str,
        selected_entity: Optional[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
    ) -> str:
        """
        构造网页检索 query。
        不能只搜用户原问题，否则网页检索容易跑偏。
        因此这里把 Task1 选出的 KG 实体名也拼进去。
        """
        parts = [self._clean_query_text(query)]

        if selected_entity and selected_entity.get("entity_name"):
            parts.append(str(selected_entity["entity_name"]))
        else:
            names = [
                item.get("entity_name", "")
                for item in kg_evidence[:2]
                if item.get("entity_name")
            ]
            parts.extend(names)

        web_query = " ".join(part for part in parts if part)
        web_query = re.sub(r"\s+", " ", web_query).strip()
        return web_query[:300] if web_query else self._clean_query_text(query)

    def _web_search(self, web_query: str) -> List[Dict[str, Any]]:
        """
        Task2 关键新增接口：网页检索。
        与 Task1 的 _image_search(image) 区分：
        - _image_search 输入 PIL Image；
        - _web_search 输入文本 query。
        """
        if self.search_pipeline is None or not web_query:
            return []

        try:
            results = self.search_pipeline(web_query, k=self.web_top_k)
        except Exception as exc:
            self._debug_task2({
                "event": "web_search_error",
                "query": web_query,
                "error": repr(exc),
            })
            return []

        return results or []

    def _build_web_evidence(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将网页检索原始结果整理成统一格式。
        兼容常见字段：
        - page_name / title / name
        - page_url / url / source_url
        - page_snippet / snippet / description / text / content / page_content
        """
        evidence = []
        seen = set()

        for idx, result in enumerate(results):
            title = self._clean_text(str(
                result.get("page_name")
                or result.get("title")
                or result.get("name")
                or ""
            ))
            url = self._clean_text(str(
                result.get("page_url")
                or result.get("url")
                or result.get("source_url")
                or ""
            ))
            snippet = self._clean_text(str(
                result.get("page_snippet")
                or result.get("snippet")
                or result.get("description")
                or result.get("summary")
                or result.get("text")
                or result.get("content")
                or result.get("page_content")
                or ""
            ))

            raw_score = float(result.get("score", 0.0) or 0.0)

            if not title and not snippet:
                continue

            key = (title.lower(), snippet[:160].lower())
            if key in seen:
                continue
            seen.add(key)

            evidence.append({
                "source": "web",
                "rank": idx + 1,
                "score": round(raw_score, 4),
                "title": title,
                "url": url,
                "snippet": snippet[:1200],
            })

        return evidence

    # ---------------------------------------------------------------------
    # Web Filtering / Ranking
    # ---------------------------------------------------------------------

    def _rank_web_evidence(
        self,
        query: str,
        web_query: str,
        web_evidence: List[Dict[str, Any]],
        selected_entity: Optional[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked = []

        for item in web_evidence:
            candidate = dict(item)
            candidate["web_rule_score"] = self._score_web_evidence(
                query=query,
                web_query=web_query,
                web_item=item,
                selected_entity=selected_entity,
                kg_evidence=kg_evidence,
            )
            if candidate["web_rule_score"] >= self.min_web_score:
                ranked.append(candidate)

        ranked.sort(key=lambda x: x.get("web_rule_score", 0.0), reverse=True)
        return ranked[: self.web_keep_top_n]

    def _score_web_evidence(
        self,
        query: str,
        web_query: str,
        web_item: Dict[str, Any],
        selected_entity: Optional[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
    ) -> float:
        """
        网页证据评分。
        目标：保留同时贴近用户问题、KG 实体和检索 query 的网页片段。
        """
        text = f"{web_item.get('title', '')} {web_item.get('snippet', '')}".lower()
        score = float(web_item.get("score", 0.0) or 0.0)

        query_tokens = self._important_tokens(query)
        web_query_tokens = self._important_tokens(web_query)
        entity_tokens = set()

        if selected_entity:
            entity_tokens |= self._important_tokens(str(selected_entity.get("entity_name", "")))
            for key, value in selected_entity.get("attributes", {}).items():
                entity_tokens |= self._important_tokens(str(key))
                entity_tokens |= self._important_tokens(str(value))

        for item in kg_evidence[:3]:
            entity_tokens |= self._important_tokens(str(item.get("entity_name", "")))

        text_tokens = self._important_tokens(text)

        if query_tokens:
            score += 0.06 * len(query_tokens & text_tokens)

        if web_query_tokens:
            score += 0.03 * len(web_query_tokens & text_tokens)

        if entity_tokens:
            score += 0.08 * len(entity_tokens & text_tokens)

        if len(str(web_item.get("snippet", ""))) < 40:
            score -= 0.08

        noise_terms = [
            "advertisement", "subscribe", "cookie", "privacy policy",
            "login", "sign up", "cart", "buy now", "sponsored"
        ]
        if any(term in text for term in noise_terms):
            score -= 0.08

        return round(max(score, 0.0), 4)

    # ---------------------------------------------------------------------
    # KG-Web Fusion
    # ---------------------------------------------------------------------

    def _fuse_multisource_evidence(
        self,
        query: str,
        selected_entity: Optional[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
        web_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        kg_candidates = []
        for item in kg_evidence[: self.rerank_top_n]:
            kg_candidates.append({
                "entity_name": item.get("entity_name", ""),
                "score": item.get("score", 0.0),
                "rule_score": item.get("rule_score", 0.0),
                "attributes": item.get("attributes", {}),
            })

        web_items = []
        for item in web_evidence[: self.web_keep_top_n]:
            web_items.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "web_rule_score": item.get("web_rule_score", 0.0),
                "url": item.get("url", ""),
            })

        return {
            "query": query,
            "selected_entity": {
                "entity_name": selected_entity.get("entity_name", "") if selected_entity else "",
                "attributes": selected_entity.get("attributes", {}) if selected_entity else {},
                "score": selected_entity.get("score", 0.0) if selected_entity else 0.0,
                "rule_score": selected_entity.get("rule_score", 0.0) if selected_entity else 0.0,
            },
            "kg_candidates": kg_candidates,
            "web_evidence": web_items,
            "policy": (
                "Image-KG evidence is primary because it is tied to visually similar images. "
                "Web evidence is auxiliary and may contain noise. "
                "Use web evidence only when it is relevant and consistent with the image-KG evidence."
            ),
        }

    # ---------------------------------------------------------------------
    # DeepSeek Answering
    # ---------------------------------------------------------------------

    def _answer_task2_with_llm(
        self,
        query: str,
        selected_entity: Optional[Dict[str, Any]],
        kg_candidates: List[Dict[str, Any]],
        web_evidence: List[Dict[str, Any]],
        fused_context: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._build_task2_answer_messages(
                    query=query,
                    selected_entity=selected_entity,
                    kg_candidates=kg_candidates,
                    web_evidence=web_evidence,
                    fused_context=fused_context,
                    history=history,
                ),
                temperature=0.1,
                max_tokens=90,
            )

            answer = response.choices[0].message.content or ""

            if selected_entity and self._is_entity_echo(answer, selected_entity):
                answer = self._repair_entity_echo(query, selected_entity, kg_candidates, history)

            if not answer.strip():
                answer = self._answer_with_rules(query, kg_candidates)

            self._debug_task2({
                "event": "task2_llm_success",
                "query": query,
                "selected_entity": selected_entity.get("entity_name") if selected_entity else None,
                "answer": answer[:240],
            })
            return answer

        except Exception as exc:
            self._debug_task2({
                "event": "task2_llm_error",
                "query": query,
                "error": repr(exc),
            })
            return self._answer_with_rules(query, kg_candidates)

    def _build_task2_answer_messages(
        self,
        query: str,
        selected_entity: Optional[Dict[str, Any]],
        kg_candidates: List[Dict[str, Any]],
        web_evidence: List[Dict[str, Any]],
        fused_context: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        selected_name = selected_entity.get("entity_name", "") if selected_entity else ""
        selected_attrs = selected_entity.get("attributes", {}) if selected_entity else {}

        selected_attr_text = self._format_attributes(selected_attrs, limit=16)
        kg_text = self._format_kg_candidates(kg_candidates[: self.rerank_top_n])
        web_text = self._format_web_evidence(web_evidence[: self.web_keep_top_n])
        history_text = self._format_history(history)

        system = (
            "You are a visual question answering assistant for a multi-source augmented task. "
            "You will receive image-KG evidence and web-search evidence. "
            "Image-KG evidence is directly retrieved from visually similar images and is the primary evidence. "
            "Web evidence is auxiliary and may contain noise. "
            "Use web evidence only when it is relevant to the question and consistent with the image-KG evidence. "
            "Do not invent unsupported facts. "
            "If the answer cannot be determined from the provided evidence, answer 'I don't know'. "
            "Answer in the same language as the user's question. "
            "Keep the answer concise, usually one sentence. "
            "Do not mention retrieval, evidence, KG, or web search in the final answer."
        )

        user = (
            f"User question:\n{query}\n\n"
            f"Selected image-KG entity:\n{selected_name}\n\n"
            f"Selected entity attributes:\n{selected_attr_text}\n\n"
            f"Other image-KG candidates:\n{kg_text}\n\n"
            f"Filtered web evidence:\n{web_text}\n\n"
            f"Conversation history, if any:\n{history_text}\n\n"
            "Evidence-use rules:\n"
            "1. Prefer selected image-KG entity and its attributes for image-specific facts.\n"
            "2. Use web evidence only to supplement missing background information or verify the selected entity.\n"
            "3. Ignore irrelevant or noisy web snippets.\n"
            "4. If web evidence conflicts with image-KG evidence, prefer image-KG evidence unless it is clearly insufficient.\n"
            "5. Directly answer the user's question. Do not explain the reasoning process.\n"
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ---------------------------------------------------------------------
    # Formatting / Utility
    # ---------------------------------------------------------------------

    def _format_attributes(self, attrs: Dict[str, Any], limit: int = 16) -> str:
        if not attrs:
            return "None"

        rows = []
        for key, value in list(attrs.items())[:limit]:
            rows.append(f"- {key}: {value}")
        return "\n".join(rows)

    def _format_kg_candidates(self, kg_candidates: List[Dict[str, Any]]) -> str:
        if not kg_candidates:
            return "None"

        rows = []
        for idx, item in enumerate(kg_candidates, start=1):
            attrs = self._format_attributes(item.get("attributes", {}), limit=6)
            attrs = attrs.replace("\n", "; ")
            rows.append(
                f"{idx}. entity={item.get('entity_name', '')}; "
                f"image_score={item.get('score', 0.0)}; "
                f"rule_score={item.get('rule_score', 0.0)}; "
                f"attributes={attrs}"
            )
        return "\n".join(rows)

    def _format_web_evidence(self, web_evidence: List[Dict[str, Any]]) -> str:
        if not web_evidence:
            return "None"

        rows = []
        for idx, item in enumerate(web_evidence, start=1):
            rows.append(
                f"{idx}. title={item.get('title', '')}; "
                f"score={item.get('web_rule_score', item.get('score', 0.0))}; "
                f"snippet={item.get('snippet', '')}"
            )
        return "\n".join(rows)

    def _important_tokens(self, text: str) -> set:
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "what", "when",
            "where", "which", "who", "why", "how", "does", "did", "was", "were",
            "are", "is", "its", "into", "about", "there", "their", "have", "has",
            "had", "can", "could", "would", "should", "will", "shall", "than",
            "then", "them", "they", "you", "your", "please", "image", "picture",
            "photo", "shown", "show", "tell", "give", "doesn", "don", "not"
        }
        tokens = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
        return {token for token in tokens if len(token) >= 3 and token not in stop_words}

    def _clean_query_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text[:240]

    def _debug_task2(self, payload: Dict[str, Any]) -> None:
        # 不记录 API key，只记录检索、过滤、融合和回答状态。
        if not self.task2_debug_path:
            return

        try:
            path = Path(self.task2_debug_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
