import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from cragmm_search.search import UnifiedSearchPipeline

try:
    from agents.Task2Agent import Task2Agent
except ImportError:
    from Task2Agent import Task2Agent


class Task3Agent(Task2Agent):
    """
    Task3 多轮问答 Agent。

    设计目标：
    1. 不改动项目骨架，直接继承 Task2Agent 的 Image-KG 与 Web 多源增强能力；
    2. 在检索前把历史对话压缩为当前轮可独立理解的问题；
    3. 在回答时显式约束上下文连续性，避免代词指代错误、前后矛盾或丢失历史信息。
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
        context_turn_limit: int = 8,
    ):
        super().__init__(
            search_pipeline=search_pipeline,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            min_score=min_score,
            model_name=model_name,
            web_top_k=web_top_k,
            web_keep_top_n=web_keep_top_n,
            min_web_score=min_web_score,
        )

        self.context_turn_limit = int(os.getenv("TASK3_CONTEXT_TURN_LIMIT", str(context_turn_limit)))
        # 默认使用可审计的规则改写；需要实验 LLM 改写时可显式设置为 0。
        self.rewrite_with_llm = os.getenv("TASK3_DISABLE_QUERY_REWRITE", "1") != "1"
        self.task3_debug_path = os.getenv("TASK3_DEBUG_PATH") or str(
            Path(__file__).resolve().parents[1] / "UI" / "outputs" / "task3" / "debug.jsonl"
        )
        # 每个会话保存首轮视觉检索形成的暂定锚点，后续追问沿用，但显式重看图片时可覆盖。
        self.visual_anchors: Dict[str, Dict[str, Any]] = {}

        self._debug_task3({
            "event": "task3_init",
            "has_client": self.client is not None,
            "model": self.model_name,
            "context_turn_limit": self.context_turn_limit,
        })

    def get_batch_size(self) -> int:
        # Task3 每个样本包含多轮历史，且可能触发 query rewrite + KG/Web + LLM 回答，先保持单样本运行。
        return 1

    def _score_candidate_by_rules(self, query: str, candidate: Dict[str, Any]) -> float:
        """补充 Task3 常见主体类别，避免品牌/地点候选压过动物、植物、船只或茶品。"""
        score = super()._score_candidate_by_rules(query, candidate)
        query_l = str(query or "").lower()
        attrs = candidate.get("attributes", {}) or {}
        name_l = str(candidate.get("entity_name", "")).lower()
        attr_keys = {str(key).lower() for key in attrs}
        taxonomy_keys = {
            "scientific_name", "genus", "species", "species_type", "species_group",
            "family", "order", "class", "phylum", "common_name", "subfamily",
        }

        category_match = False
        if any(term in query_l for term in ["animal", "lifespan", "phylum", "squid", "octopus"]):
            animal_names = ["octopus", "hapalochlaena", "eledone", "japetella", "ocythoe", "squid", "megalodon"]
            category_match = any(term in name_l for term in animal_names) or bool(attr_keys & taxonomy_keys)
        elif any(term in query_l for term in ["plant", "leaves", "flower", "ivy", "droopy"]):
            plant_keys = taxonomy_keys | {"leaf", "leaves", "flower", "growth_habit", "plant_type"}
            category_match = "ivy" in name_l or bool(attr_keys & plant_keys)
        elif any(term in query_l for term in ["boat", "rows", "rower", "gondola"]):
            category_match = any(term in name_l for term in ["boat", "ship", "vessel", "gondola", "canoe", "ferry"])
        elif any(term in query_l for term in ["tea", "steepster", "caffeine", "steeping chart"]):
            tea_keys = {"tea_type", "caffeine", "brand", "company", "steeping_time"}
            category_match = any(term in name_l for term in ["tea", "genmaicha"]) or bool(attr_keys & tea_keys)

        if category_match:
            # 只使用实体名和结构化字段判断类别，不能让品牌 description 中偶然出现的 animal 一词获加分。
            # 类别只做轻量平局裁决，不能压过原始图像相似度。
            score += 0.03
        return round(score, 4)

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        if not (len(queries) == len(images) == len(message_histories)):
            raise ValueError(
                "Task3 批量输入长度不一致："
                f"queries={len(queries)}, images={len(images)}, histories={len(message_histories)}"
            )
        responses = []

        for batch_index, (query, image, history) in enumerate(zip(queries, images, message_histories)):
            trace = self._trace_context(batch_index)
            session_id = str(trace.get("session_id") or "")
            anchor = self.visual_anchors.get(session_id, {}) if session_id else {}
            context = self._build_context_state(history, anchor)
            contextual_query = self._rewrite_query_with_context(query, context)

            # 1. 数据集会在每一轮重复传入同一张图片。首轮或明确再次询问图片时才检索图片，
            # 普通知识追问沿用会话上下文，避免相同图片在不同问题权重下跳到无关实体。
            use_image = self._should_use_image(query, image, context)
            if use_image:
                raw_image_results = self._image_search(image)
                kg_evidence = self._build_evidence(raw_image_results)
            else:
                raw_image_results = []
                # 普通知识追问沿用首轮视觉候选，避免每轮重搜造成主体漂移，也避免 KG 为空。
                kg_evidence = [dict(item) for item in anchor.get("candidates", [])]

            legacy_ranked = self._rank_candidates_by_rules(contextual_query, kg_evidence)
            legacy_ranked = self._rerank_kg_with_context(legacy_ranked, contextual_query, context)
            visual_result = self._prepare_visual_evidence(
                contextual_query,
                image,
                history,
                legacy_ranked,
                # 每轮问题的指向可能变化，按当前问题生成锚点；相同请求由视觉管线缓存去重。
                cached_anchor=None,
                refresh_anchor=True,
                trace=trace,
            )
            ranked_kg = visual_result["candidates"]
            initial_support = self._select_supporting_entities(contextual_query, ranked_kg)
            initial_entity = visual_result.get("selected_entity") or (initial_support[0] if initial_support else (ranked_kg[0] if ranked_kg else None))

            # 2. 多轮问题经常含有 it/that/this 等指代词，web query 使用改写后的独立问题。
            web_query = self._build_task3_web_query(contextual_query, context, initial_entity, initial_support or ranked_kg)
            broad_web = self._build_web_evidence(self._web_search(contextual_query), source="broad")
            entity_web = self._build_web_evidence(
                self._web_search(web_query) if web_query.lower() != contextual_query.lower() else [],
                source="entity_directed",
            )
            web_evidence = self._merge_web_results(broad_web, entity_web)
            if visual_result.get("fallback_used"):
                ranked_kg = self._rerank_kg_with_web(ranked_kg, broad_web)
            support_entities = self._select_supporting_entities(contextual_query, ranked_kg)
            selected_entity = visual_result.get("selected_entity") or self._select_entity(contextual_query, support_entities or ranked_kg)

            if session_id and selected_entity:
                anchor = self._update_visual_anchor(
                    session_id,
                    selected_entity,
                    support_entities or ranked_kg,
                    trace.get("turn_idx"),
                    visual_result.get("anchor", {}),
                )
                context["visual_anchor"] = anchor
            ranked_web = self._rank_web_evidence(
                query=contextual_query,
                web_query=web_query,
                web_evidence=web_evidence,
                selected_entity=selected_entity,
                kg_evidence=support_entities or ranked_kg,
            )

            # 3. KG-Web 融合仍复用 Task2，只是 query 使用上下文改写后的版本。
            fused_context = self._fuse_multisource_evidence(
                query=contextual_query,
                selected_entity=selected_entity,
                kg_evidence=support_entities or ranked_kg,
                web_evidence=ranked_web,
            )

            if self.client is not None:
                answer = self._answer_task3_with_llm(
                    original_query=query,
                    contextual_query=contextual_query,
                    context=context,
                    selected_entity=selected_entity,
                    kg_candidates=support_entities or ranked_kg,
                    web_evidence=ranked_web,
                    fused_context=fused_context,
                )
            else:
                # 无 API key 时保留可运行兜底，方便做环境 smoke test。
                answer = self._answer_task3_without_llm(contextual_query, support_entities or ranked_kg)
                answer = answer or self._answer_with_heuristic_sentence(contextual_query, support_entities or ranked_kg)
                answer = answer or self._answer_with_rules(contextual_query, support_entities or ranked_kg)

            answer = self._finalize_answer(answer)
            responses.append(answer)

            self._debug_task3({
                "event": "task3_query",
                "query": query,
                "session_id": session_id,
                "interaction_id": trace.get("interaction_id"),
                "turn_idx": trace.get("turn_idx"),
                "contextual_query": contextual_query,
                "history_turns": len(context.get("turns", [])),
                "use_image": use_image,
                "selected_entity": selected_entity.get("entity_name") if selected_entity else None,
                "anchor_entity": (anchor.get("selected_entity") or {}).get("entity_name") if anchor else None,
                "qwen_anchor_subject": (visual_result.get("anchor") or {}).get("primary_subject"),
                "qwen_confidence": (visual_result.get("rerank") or {}).get("confidence", 0.0),
                "vision_fallback": visual_result.get("fallback_used"),
                "vision_fallback_reason": visual_result.get("fallback_reason"),
                "image_score": anchor.get("image_score") if anchor else None,
                "image_margin": anchor.get("image_margin") if anchor else None,
                "kg_count": len(kg_evidence),
                "kg_score_components": [
                    {
                        "entity": item.get("entity_name"),
                        "image_score": item.get("score"),
                        "context_support_score": item.get("context_support_score", 0.0),
                        "web_support_score": item.get("web_support_score", 0.0),
                        "final_score": item.get("rule_score"),
                    }
                    for item in ranked_kg[: self.rerank_top_n]
                ],
                "web_query": web_query,
                "web_count": len(web_evidence),
                "ranked_web_titles": [item.get("title") for item in ranked_web[: self.web_keep_top_n]],
                "answer": answer[:260],
                "has_client": self.client is not None,
            })

        return responses

    def _update_visual_anchor(
        self,
        session_id: str,
        selected_entity: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        turn_idx: Any,
        qwen_anchor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """写入或纠正会话视觉锚点；显式重新看图得到的新证据可以覆盖旧锚点。"""
        anchor_candidates = [dict(item) for item in candidates[: self.rerank_top_n]]
        scores = [float(item.get("score", 0.0) or 0.0) for item in anchor_candidates]
        anchor = {
            "selected_entity": dict(selected_entity),
            "candidates": anchor_candidates,
            "image_score": scores[0] if scores else 0.0,
            "image_margin": (scores[0] - scores[1]) if len(scores) > 1 else 0.0,
            "turn_idx": turn_idx,
            "qwen_anchor": qwen_anchor or {},
        }
        self.visual_anchors[session_id] = anchor
        return anchor

    # ------------------------------------------------------------------
    # 上下文优化
    # ------------------------------------------------------------------

    def _build_context_state(
        self,
        history: List[Dict[str, Any]],
        visual_anchor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 将官方 evaluator 传入的 message_history 压缩成短上下文，避免把过长历史直接塞进检索 query。
        history = history or []
        turns = []
        trusted_turns = []
        for msg in history[-self.context_turn_limit:]:
            role = str(msg.get("role", "")).strip() or "unknown"
            content = self._clean_query_text(msg.get("content", ""))
            if content:
                item = {"role": role, "content": content[:500]}
                turns.append(item)
                # IDK 不能提供实体信息，把它拼进下一轮检索会形成连续污染。
                if role != "assistant" or not self._is_unknown_answer(content):
                    trusted_turns.append(item)

        last_user = next((item["content"] for item in reversed(trusted_turns) if item["role"] == "user"), "")
        last_assistant = next((item["content"] for item in reversed(trusted_turns) if item["role"] == "assistant"), "")
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in trusted_turns) if trusted_turns else "None"
        user_history_text = " ".join(item["content"] for item in trusted_turns if item["role"] == "user")

        anchor_name = str((visual_anchor or {}).get("selected_entity", {}).get("entity_name", "")).strip()
        recent_entities = self._extract_recent_entities(user_history_text)
        if anchor_name and anchor_name.lower() not in {item.lower() for item in recent_entities}:
            recent_entities.insert(0, anchor_name)
        return {
            "turns": turns,
            "trusted_turns": trusted_turns,
            "history_text": history_text,
            "last_user_question": last_user,
            "last_assistant_answer": last_assistant,
            "user_history_text": user_history_text,
            "recent_entities": recent_entities,
            "visual_anchor": visual_anchor or {},
            "has_history": bool(trusted_turns),
        }

    def _rerank_kg_with_context(
        self,
        ranked_kg: List[Dict[str, Any]],
        contextual_query: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """使用当前问题和用户历史给 KG 候选加分，不盲信旧 assistant 答案。"""
        context_text = f"{contextual_query} {context.get('user_history_text', '')}".lower()
        context_tokens = self._important_tokens(context_text)
        reranked = []
        for item in ranked_kg:
            candidate = dict(item)
            name = str(candidate.get("entity_name", "")).strip().lower()
            name_tokens = self._important_tokens(name)
            bonus = 0.0
            if len(name) >= 4 and name in context_text:
                bonus += 0.45
            if name_tokens:
                bonus += min(0.18, 0.06 * len(name_tokens & context_tokens))
            candidate["context_support_score"] = round(bonus, 4)
            candidate["rule_score"] = round(float(candidate.get("rule_score", 0.0) or 0.0) + bonus, 4)
            reranked.append(candidate)
        return sorted(reranked, key=lambda item: item.get("rule_score", 0.0), reverse=True)

    def _rewrite_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        # 当前问题本身完整时直接使用；只有多轮上下文存在时才做改写。
        query = self._clean_query_text(query)
        if not context.get("has_history"):
            return query

        if self.client is not None and self.rewrite_with_llm:
            rewritten = self._rewrite_query_with_llm(query, context)
            if rewritten:
                return rewritten

        # 规则兜底：只拼接可信历史；绝不把 I don't know 放入检索问题。
        if self._looks_like_followup(query):
            context_bits = [context.get("last_user_question", ""), context.get("last_assistant_answer", "")]
            context_text = " ".join(bit for bit in context_bits if bit)
            return self._clean_query_text(f"{context_text} Follow-up: {query}")
        return query

    def _rewrite_query_with_llm(self, query: str, context: Dict[str, Any]) -> str:
        # DeepSeek 只负责把当前轮改写成独立检索问题，不负责生成最终答案。
        try:
            rewritten = self._call_llm(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是多轮问答的检索问题改写器。请根据历史对话，把当前问题改写成一个可独立检索的问题。"
                            "如果当前问题已经完整，只做轻微清理。不要回答问题，不要解释，只输出改写后的问题。"
                            "历史中的 assistant 回答只是模型旧答案，可能不正确；优先使用连续的 user 问题提供的实体线索。"
                            "保留实体名、时间、数量、指代对象和图片中的目标对象，不要在结果中加入 I don't know。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"历史对话：\n{context.get('history_text', 'None')}\n\n"
                            f"当前问题：\n{query}\n\n"
                            "请输出一行独立检索问题："
                        ),
                    },
                ],
                max_tokens=192,
                purpose="task3_query_rewrite",
            )
            rewritten = self._clean_query_text(rewritten.strip().strip('"').strip("'"))
            if rewritten and "don't know" not in rewritten.lower():
                self._debug_task3({"event": "task3_query_rewrite", "query": query, "rewritten": rewritten})
                return rewritten
        except Exception as exc:
            self._debug_task3({"event": "task3_query_rewrite_error", "query": query, "error": repr(exc)})
        return ""

    def _build_task3_web_query(
        self,
        contextual_query: str,
        context: Dict[str, Any],
        selected_entity: Optional[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
    ) -> str:
        # 在 Task2 web query 基础上加入最近实体，增强指代问题的召回。
        base_query = contextual_query
        recent_entities = context.get("recent_entities", [])[:3]
        if recent_entities and self._looks_like_followup(contextual_query):
            base_query = f"{contextual_query} {' '.join(recent_entities)}"
        return self._build_web_query(base_query, selected_entity, kg_evidence)

    # ------------------------------------------------------------------
    # 回答生成
    # ------------------------------------------------------------------

    def _answer_task3_without_llm(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """无 API 时不注入特定实体答案，交由通用 KG 规则继续处理。"""
        return ""

    def _answer_task3_with_llm(
        self,
        original_query: str,
        contextual_query: str,
        context: Dict[str, Any],
        selected_entity: Optional[Dict[str, Any]],
        kg_candidates: List[Dict[str, Any]],
        web_evidence: List[Dict[str, Any]],
        fused_context: Dict[str, Any],
    ) -> str:
        # Task3 回答接口：强调历史一致性，同时沿用 Task1/Task2 的完整句质量闸门。
        try:
            answer = self._call_llm(
                self._build_task3_answer_messages_clean(
                    original_query=original_query,
                    contextual_query=contextual_query,
                    context=context,
                    selected_entity=selected_entity,
                    kg_candidates=kg_candidates,
                    web_evidence=web_evidence,
                    fused_context=fused_context,
                ),
                max_tokens=512,
                purpose="task3_answer",
            )
            raw_answer = answer
            answer, retry = self._maybe_evidence_retry(
                query=contextual_query, initial_answer=answer, selected=selected_entity,
                kg_candidates=kg_candidates, web_evidence=web_evidence,
                history=context.get("turns", []), purpose="task3_evidence_retry",
            )
            quality_branches = []

            if self._needs_sentence_rewrite(answer, contextual_query, kg_candidates) or not self._answer_addresses_current_question(answer, original_query, context):
                quality_branches.append("llm_rewrite")
                answer = self._rewrite_task3_as_sentence(original_query, contextual_query, answer, context, kg_candidates, web_evidence) or answer

            if self._needs_sentence_rewrite(answer, contextual_query, kg_candidates) or not self._answer_addresses_current_question(answer, original_query, context):
                quality_branches.append("heuristic_fallback")
                answer = self._answer_with_heuristic_sentence(contextual_query, kg_candidates) or answer

            if self._needs_sentence_rewrite(answer, contextual_query, kg_candidates) or not self._answer_addresses_current_question(answer, original_query, context):
                quality_branches.append("forced_idk")
                answer = "I don't know."

            self._debug_task3({
                "event": "task3_llm_success",
                "query": original_query,
                "contextual_query": contextual_query,
                "raw_answer": raw_answer[:260],
                "evidence_retry": retry,
                "answer": answer[:260],
                "quality_branches": quality_branches,
                "is_idk": "i don't know" in answer.lower(),
                "idk_reason": (
                    "empty_llm_content" if not raw_answer.strip()
                    else "llm_returned_idk" if "i don't know" in raw_answer.lower()
                    else "quality_gate" if "forced_idk" in quality_branches
                    else ""
                ),
            })
            return answer

        except Exception as exc:
            self._debug_task3({"event": "task3_llm_error", "query": original_query, "error": repr(exc)})
            return self._answer_with_heuristic_sentence(contextual_query, kg_candidates) or self._answer_with_rules(contextual_query, kg_candidates)

    def _build_task3_answer_messages(
        self,
        original_query: str,
        contextual_query: str,
        context: Dict[str, Any],
        selected_entity: Optional[Dict[str, Any]],
        kg_candidates: List[Dict[str, Any]],
        web_evidence: List[Dict[str, Any]],
        fused_context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        # 中文 prompt 更稳定；最终回答语言按用户问题保持，英文问题输出英文答案。
        kg_text = self._format_kg_candidates(kg_candidates[: self.rerank_top_n])
        web_text = self._format_web_evidence(web_evidence[: self.web_keep_top_n])
        selected_name = selected_entity.get("entity_name") if selected_entity else "None"

        system = (
            "你是 CRAG-MM Task3 多轮视觉问答助手。系统会提供历史对话、当前问题、上下文改写后的检索问题、"
            "Image-KG 候选实体和 Web 证据。你的重点是保持多轮上下文连续：正确理解 it/this/that/they 等指代，"
            "历史中的 assistant 内容是旧模型答案，不保证正确；用户连续提问中出现的新实体线索优先级更高。"
            "不要机械复述上一轮答案；当前轮若问的是属性、人物、地点、数量或原因，必须回答对应属性。"
            "如果历史答案与当前问题或网页事实冲突，应使用更具体的新线索修正实体并回答当前问题。"
            "请直接回答当前轮用户真正想问的问题。不要只输出实体名、标题、车型名、建筑名或逗号短语。"
            "网页标题或片段能直接回答当前问题时可以据此作答，不要仅因图片候选不一致就回答不知道。"
            "只有历史、KG 和 Web 都没有相关信息时才输出完整句 I don't know.。用户用英文问就用英文答；最终答案最多两句话。"
            "不要提到 KG、网页、检索、候选、证据编号或推理过程。"
        )

        user = (
            f"历史对话：\n{context.get('history_text', 'None')}\n\n"
            f"当前原问题：\n{original_query}\n\n"
            f"上下文改写后的检索问题：\n{contextual_query}\n\n"
            f"当前优先实体：{selected_name}\n\n"
            f"Image-KG 候选实体与属性：\n{kg_text}\n\n"
            f"筛选后的 Web 证据：\n{web_text}\n\n"
            "请结合历史上下文和可用信息，输出一个自然、完整、与上下文一致的最终答案："
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_task3_answer_messages_clean(
        self,
        original_query: str,
        contextual_query: str,
        context: Dict[str, Any],
        selected_entity: Optional[Dict[str, Any]],
        kg_candidates: List[Dict[str, Any]],
        web_evidence: List[Dict[str, Any]],
        fused_context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """构造 UTF-8 中文 Task3 Prompt，显式提供会话锚点并允许新证据纠错。"""
        anchor = context.get("visual_anchor", {}) or {}
        anchor_name = str((anchor.get("selected_entity") or {}).get("entity_name", "未确定"))
        qwen_anchor = anchor.get("qwen_anchor", {}) or {}
        selected_name = selected_entity.get("entity_name", "未确定") if selected_entity else "未确定"
        system = (
            "你是 CRAG-MM Task3 多轮视觉问答助手。每一轮都必须回答当前问题，并保持同一会话中的图片主体一致。"
            "视觉锚点是首轮图片检索得到的暂定主体，不是绝对真相；若当前问题或新的直接证据与它冲突，应纠正锚点。"
            "历史 assistant 回答可能错误，只能作为弱线索，不能覆盖用户问题、视觉候选和直接网页事实。"
            "正确解析 it、this、that、the company、the city 等指代。问题询问属性时必须给出该属性，不能重复上一轮主体名称。"
            "只有历史、视觉候选和网页证据都不能支持答案时才回答 I don't know."
            "英文问题使用英文完整句回答，最多两句，不得提及检索、KG、候选、网页或推理过程。"
        )
        user = (
            f"历史对话：\n{context.get('history_text', 'None')}\n\n"
            f"当前问题：{original_query}\n"
            f"用于检索的上下文问题：{contextual_query}\n\n"
            f"会话视觉锚点：{anchor_name}\n"
            f"Qwen 当前问题目标：{qwen_anchor.get('question_target', '未确定')}\n"
            f"Qwen 图片主体：{qwen_anchor.get('primary_subject', '未确定')}\n"
            f"Qwen 可见文字：{qwen_anchor.get('visible_text', [])}\n"
            f"本轮暂定实体：{selected_name}\n\n"
            f"视觉候选与属性：\n{self._format_kg_candidates(kg_candidates[:self.rerank_top_n])}\n\n"
            f"网页补充证据：\n{self._format_web_evidence(web_evidence[:self.web_keep_top_n])}\n\n"
            "请直接输出当前问题的最终英文答案。"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _rewrite_task3_as_sentence(
        self,
        original_query: str,
        contextual_query: str,
        bad_answer: str,
        context: Dict[str, Any],
        kg_candidates: List[Dict[str, Any]],
        web_evidence: List[Dict[str, Any]],
    ) -> str:
        # 二次改写接口：当模型输出短语/实体名时，强制生成与多轮上下文一致的完整句。
        try:
            rewritten = self._call_llm(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是答案改写器。上一版答案可能只是实体名、短语或空串。"
                            "请根据历史对话和证据，把它改写为直接回答当前问题的完整自然句。"
                            "英文问题输出英文答案；最多两句话；证据不足输出完整句 I don't know."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"历史对话：\n{context.get('history_text', 'None')}\n\n"
                            f"当前原问题：{original_query}\n"
                            f"上下文改写问题：{contextual_query}\n"
                            f"不合格答案：{bad_answer}\n\n"
                            f"KG 候选：\n{self._format_kg_candidates(kg_candidates[: self.rerank_top_n])}\n\n"
                            f"Web 证据：\n{self._format_web_evidence(web_evidence[: self.web_keep_top_n])}\n\n"
                            "请只输出最终答案："
                        ),
                    },
                ],
                max_tokens=192,
                purpose="task3_sentence_rewrite",
            )
            self._debug_task3({"event": "task3_sentence_rewrite", "query": original_query, "bad_answer": bad_answer[:160], "rewritten": rewritten[:220]})
            return rewritten
        except Exception as exc:
            self._debug_task3({"event": "task3_sentence_rewrite_error", "query": original_query, "error": repr(exc)})
            return ""

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------

    def _is_unknown_answer(self, answer: str) -> bool:
        """判断历史回答是否是不提供实体信息的 IDK。"""
        text = re.sub(r"\s+", " ", str(answer or "")).strip().lower()
        return not text or "i don't know" in text or "i don’t know" in text or text in {"unknown", "none"}

    def _should_use_image(self, query: str, image: Any, context: Dict[str, Any]) -> bool:
        """Task3 首轮建立视觉锚点，后续只在问题明确重新指向图片时再次检索。"""
        if not self._has_usable_image(image):
            return False
        if not context.get("has_history"):
            return True
        query_l = str(query or "").lower()
        explicit_image_terms = [
            "in the image", "in this image", "in the picture", "in this picture",
            "shown here", "on the chart", "on the package", "top left", "bottom left",
            "top right", "bottom right", "what color", "which color", "appearance",
            "look like", "visible", "shown", "pictured",
        ]
        return any(term in query_l for term in explicit_image_terms)

    def _answer_addresses_current_question(
        self,
        answer: str,
        query: str,
        context: Dict[str, Any],
    ) -> bool:
        """阻止 Agent 在追问中机械重复上一轮身份答案。"""
        if self._is_unknown_answer(answer):
            return True
        normalized_answer = re.sub(r"[^a-z0-9]+", " ", str(answer).lower()).strip()
        previous = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(context.get("last_assistant_answer", "")).lower(),
        ).strip()
        current_query = re.sub(r"[^a-z0-9]+", " ", str(query).lower()).strip()
        previous_query = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(context.get("last_user_question", "")).lower(),
        ).strip()
        if previous and normalized_answer == previous and current_query != previous_query:
            return False

        # 数值追问需要实际给出数字或常见数量词。
        if any(term in current_query for term in ["how many", "how much", "how long"]):
            if not re.search(r"\b\d+(?:\.\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|dozen)\b", normalized_answer):
                return False
        return True

    def _has_usable_image(self, image: Any) -> bool:
        # Task3 后续轮次可能不依赖图像；这里做宽松检测，避免空图导致检索报错。
        return isinstance(image, Image.Image) and image.size[0] > 1 and image.size[1] > 1

    def _looks_like_followup(self, query: str) -> bool:
        text = str(query or "").lower()
        tokens = set(re.findall(r"[a-z0-9']+", text))
        pronouns = {
            "it", "its", "this", "that", "these", "those", "they", "them", "their",
            "he", "she", "his", "her", "also", "same", "previous", "earlier", "there",
        }
        followup_phrases = ["what about", "how about", "and what", "the one"]
        return bool(tokens & pronouns) or any(phrase in text for phrase in followup_phrases)

    def _extract_recent_entities(self, text: str) -> List[str]:
        # 从历史中粗略抽取实体/年份/型号，供检索 query 兜底使用，不参与最终回答判定。
        if not text:
            return []
        patterns = [
            r"\b[A-Z][A-Za-z0-9]+(?:[- ][A-Z]?[A-Za-z0-9]+){0,3}\b",
            r"\b\d{4}\b",
            r"\b[A-Z]{2,}\b",
        ]
        seen = set()
        entities = []
        for pattern in patterns:
            for match in re.findall(pattern, text):
                value = re.sub(r"\s+", " ", str(match)).strip()
                if len(value) < 2:
                    continue
                key = value.lower()
                if key in seen or key in {"user", "assistant", "none"}:
                    continue
                seen.add(key)
                entities.append(value)
                if len(entities) >= 8:
                    return entities
        return entities

    def _debug_task3(self, payload: Dict[str, Any]) -> None:
        # 不记录 API key，只记录上下文改写、检索和回答状态，便于本地调试 Task3。
        if not self.task3_debug_path:
            return
        try:
            path = Path(self.task3_debug_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
