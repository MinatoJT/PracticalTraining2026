from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from typing import Any, Dict, List

from PIL import Image

from ..config import AgentConfig
from ..conversation import fallback_standalone_question, format_history, infer_question_type
from ..schemas import AnswerDecision, EvidenceItem, QueryPlan, RerankDecision, clamp01, extract_json_object
from ..tracing import TraceWriter
from .qwen_local import LocalQwen3VLBackend


class QwenVisionProvider:
    """Qwen visual planner/reranker with API-first and optional local BF16 backends."""

    def __init__(self, config: AgentConfig, trace: TraceWriter):
        self.config = config
        self.trace = trace
        self.backend = config.qwen_backend if config.qwen_backend in {"api", "local"} else "api"
        self._client = None
        self._local = None
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "backend": self.backend,
            "planner_calls": 0,
            "rerank_calls": 0,
            "audit_calls": 0,
            "answer_verification_calls": 0,
            "errors": 0,
            "fallbacks": 0,
            "latency_total": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        if self.backend == "local":
            self._local = LocalQwen3VLBackend.shared(config.qwen_local_model, config.qwen_max_image_edge)
        else:
            self._client = self._build_api_client()

    def _build_api_client(self):
        key = (os.getenv("QWEN_VL_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip().strip('"').strip("'")
        base_url = os.getenv(
            "QWEN_VL_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip().rstrip("/")
        if not key or not base_url.startswith(("http://", "https://")):
            return None
        try:
            from openai import OpenAI

            return OpenAI(api_key=key, base_url=base_url, timeout=self.config.qwen_timeout, max_retries=0)
        except Exception:
            return None

    @property
    def available(self) -> bool:
        if self.backend == "local":
            return bool(self._local and self._local.available)
        return self._client is not None

    def _image_data_url(self, image: Image.Image) -> str:
        prepared = image.convert("RGB").copy()
        prepared.thumbnail((self.config.qwen_max_image_edge, self.config.qwen_max_image_edge))
        buffer = io.BytesIO()
        prepared.save(buffer, format="JPEG", quality=90, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _request_api(self, image: Image.Image, prompt: str, model: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                {"type": "text", "text": prompt},
            ],
        }]
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.config.qwen_max_tokens,
            "response_format": {"type": "json_object"},
        }
        normalized_model = model.lower()
        if "qwen3-vl" in normalized_model and "realtime" not in normalized_model:
            kwargs["extra_body"] = {"enable_thinking": False}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            text = repr(exc).lower()
            if "response_format" not in text and "unsupported" not in text and "unknown parameter" not in text:
                raise
            kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**kwargs)
        choices = list(getattr(response, "choices", None) or [])
        if not choices:
            raise RuntimeError("qwen_choices_empty")
        content = str(getattr(choices[0].message, "content", None) or "").strip()
        if not content:
            raise RuntimeError("qwen_content_empty")
        usage = getattr(response, "usage", None)
        with self._lock:
            self._stats["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._stats["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        return content

    def _request(self, image: Image.Image, prompt: str, model: str, purpose: str) -> Dict[str, Any]:
        if not isinstance(image, Image.Image) or not self.available:
            with self._lock:
                self._stats["fallbacks"] += 1
            return {}
        started = time.perf_counter()
        try:
            if self.backend == "local":
                content = self._local.generate(image, prompt, self.config.qwen_max_tokens)
            else:
                content = self._request_api(image, prompt, model)
            parsed = extract_json_object(content)
            if not parsed:
                raise RuntimeError("qwen_invalid_json")
            return parsed
        except Exception as exc:
            with self._lock:
                self._stats["errors"] += 1
                self._stats["fallbacks"] += 1
            self.trace.write("qwen_error", purpose=purpose, model=model, error=type(exc).__name__)
            return {}
        finally:
            with self._lock:
                self._stats["latency_total"] += time.perf_counter() - started

    def plan(self, image: Image.Image, question: str, history: List[Dict[str, Any]], web_enabled: bool) -> QueryPlan:
        fallback = fallback_standalone_question(question, history, self.config.max_history_messages)
        search_instruction = (
            "Produce 2 or 3 short, diverse web search queries. Every query must target the final requested attribute, "
            "not merely an intermediate entity. Use one fully resolved question, one exact subject plus attribute query, "
            "and one primary/official-source or second-hop query. For relation chains, identify the likely intermediate "
            "entity and ask for the final attribute in the same query. Preserve edition, year, trim, submodel, and region "
            "qualifiers; do not silently broaden a submodel into its product family."
            if web_enabled
            else "Do not produce web search queries; this task may use only image-KG evidence."
        )
        prompt = (
            "You are the independent visual-anchor stage of a multimodal RAG system. Inspect the image closely and "
            "name the most specific subject needed to answer the current question. Do not let later retrieval candidate "
            "titles bias this judgment. Resolve references using user history, but old assistant answers may be wrong. "
            "For vehicles, food, landmarks, products, plants, or machines, distinguish the exact subtype/model when "
            "visible; record useful OCR text. If exact identity is ambiguous, give the best-supported category rather "
            "than inventing a model. A bare follow-up such as 'what is that?' normally asks about the main concept in "
            "the immediately preceding user question, not the original pictured object. Resolve references by semantic "
            "type: 'stand for' requires a name, acronym, or symbol antecedent, not an unrelated numeric quantity. For "
            "a singular definite reference after several same-type entities were listed, use the last-listed compatible "
            "entity unless the current question explicitly names an earlier one. For an address-to-building relation, "
            "include a quoted exact-address query with terms such as office tower or building name. For "
            "spatial, highlighted, arrowed, or 'main part' questions, inspect the requested region and diagram markings; "
            "do not substitute the concept that is most famous or generally considered central. If the question asks "
            "for a creator, identify the exact work and likely creator when visually supportable, not only its broad genre. "
            f"{search_instruction} Do not answer the question. Return one JSON object only.\n"
            "Fields: standalone_question, question_type, visual_target, visible_text (string array), "
            "search_queries (string array), requires_web (boolean), requires_visual_recheck (boolean), confidence (0..1).\n"
            f"History:\n{format_history(history, self.config.max_history_messages)}\n"
            f"Current question: {question}"
        )
        with self._lock:
            self._stats["planner_calls"] += 1
        raw = self._request(image, prompt, self.config.qwen_anchor_model, "planner")
        if not raw:
            return QueryPlan(
                original_question=question,
                standalone_question=fallback,
                question_type=infer_question_type(question),
                search_queries=[fallback] if web_enabled and fallback else [],
                requires_web=web_enabled,
            )
        plan = QueryPlan.from_dict(raw, question, web_enabled=web_enabled)
        plan.search_queries = plan.search_queries[: self.config.max_search_queries] if web_enabled else []
        self.trace.write(
            "query_plan",
            question=question,
            standalone_question=plan.standalone_question,
            question_type=plan.question_type,
            visual_target=plan.visual_target,
            search_queries=plan.search_queries,
            confidence=plan.confidence,
        )
        return plan

    def rerank(
        self,
        image: Image.Image,
        plan: QueryPlan,
        evidence: List[EvidenceItem],
        history: List[Dict[str, Any]],
    ) -> RerankDecision:
        shortlist = evidence[: self.config.rerank_shortlist]
        if not shortlist:
            return RerankDecision(answerable=False, reason="no_evidence")
        compact = [item.prompt_dict(max_text=520, max_attributes=6) for item in shortlist]
        prompt = (
            "You are the visual evidence reranker in a multimodal RAG system. Judge each candidate against the "
            "original image and the standalone question. Image-KG candidates primarily establish visual identity; "
            "web candidates provide facts but may be noisy or inherit a wrong entity query. Distinguish a real object "
            "from an object depicted on a poster, screen, package, artwork, or chart. Candidate titles are hints, not "
            "a closed set: keep the independent visual anchor when retrieval is wrong. Set subject to the most specific "
            "visually supported identity. Here answerable means that the visual subject is clear enough to attempt the "
            "question; it does not require the candidates to contain the final fact. Calibrate confidence to the exact "
            "identity required: use confidence at or below 0.45 when only a broad category or generic artwork title is "
            "known. Return JSON only.\n"
            "Perform question-directed OCR: inspect labels, tables, captions, menus, and nearby numbers rather than "
            "summarizing the whole image. Put only text or facts directly visible in the pixels into visual_facts; do "
            "not add outside knowledge there. Fields: subject (string), confidence (0..1), answerable (boolean), "
            "visual_facts (string array), missing_information (string), "
            "reason (short string), ranked=[{id,relevance,supports_answer,reason}]. Relevance is 0..1. Return only the "
            "10 most relevant candidates in ranked, in descending order, and use only IDs supplied.\n"
            f"Question: {plan.standalone_question}\n"
            f"Question type: {plan.question_type}\n"
            f"Visual target from planning: {plan.visual_target or 'unknown'}\n"
            f"Visible text: {json.dumps(plan.visible_text, ensure_ascii=False)}\n"
            f"Conversation history (assistant text is unverified):\n{format_history(history, self.config.max_history_messages)}\n"
            f"Candidates: {json.dumps(compact, ensure_ascii=False)}"
        )
        with self._lock:
            self._stats["rerank_calls"] += 1
        raw = self._request(image, prompt, self.config.qwen_rerank_model, "rerank")
        needs_audit = bool(raw) and clamp01(raw.get("confidence")) < 0.55
        if needs_audit and self.config.qwen_anchor_model != self.config.qwen_rerank_model:
            with self._lock:
                self._stats["audit_calls"] += 1
            audit = self._request(
                image,
                "High-precision audit: independently recheck the exact visual identity and correct the prior low-"
                "confidence judgment. Do not force a candidate match.\n" + prompt,
                self.config.qwen_anchor_model,
                "rerank_audit",
            )
            if audit and clamp01(audit.get("confidence")) > clamp01(raw.get("confidence")):
                raw = audit
        if not raw:
            return RerankDecision(answerable=None, reason="provider_fallback")
        valid_ids = {item.eid for item in shortlist}
        scores: Dict[str, float] = {}
        for item in raw.get("ranked", []) or []:
            eid = str(item.get("id") or "").strip()
            if eid in valid_ids:
                scores[eid] = clamp01(item.get("relevance"))
        return RerankDecision(
            answerable=raw.get("answerable") if isinstance(raw.get("answerable"), bool) else None,
            subject=str(raw.get("subject") or "").strip(),
            confidence=clamp01(raw.get("confidence")),
            ranked_scores=scores,
            visual_facts=[
                " ".join(str(item).split())[:500]
                for item in (raw.get("visual_facts", []) or [])
                if str(item).strip()
            ][:12],
            missing_information=str(raw.get("missing_information") or "").strip(),
            reason=str(raw.get("reason") or "").strip(),
        )

    def verify_answer(
        self,
        image: Image.Image,
        plan: QueryPlan,
        evidence: List[EvidenceItem],
        history: List[Dict[str, Any]],
        subject: str,
        candidate: AnswerDecision,
    ) -> AnswerDecision:
        """Use the fixed high-precision Qwen anchor model as final adjudicator."""
        if not isinstance(image, Image.Image) or not self.available:
            return candidate
        compact = [item.prompt_dict(max_text=700, max_attributes=8) for item in evidence[: self.config.answer_evidence_limit]]
        prompt = (
            "You are the fixed high-precision final adjudicator in a multimodal RAG pipeline. This is an intentional "
            "verification stage, not a provider fallback. Inspect the image, current question, dialogue, proposed answer, "
            "and every evidence item. Earlier assistant messages and the proposed answer may be wrong. Resolve pronouns "
            "by semantic type and conversation topic. Verify the exact pictured entity, edition, product variant, company, "
            "person, requested attribute, historical scope, unit, serving basis, and comparison direction. Prefer evidence "
            "that directly matches the entity and attribute; do not trust rank alone. Preserve a fully correct proposed "
            "answer. Correct it only when image or evidence clearly supports the correction. A list or time-range question "
            "must include all major supported items; a side-effect or consequence question needs the requested specifics, "
            "not only Yes. If sources conflict or the exact fact is unsupported, answer I don't know. Keep the answer "
            "concise and do not mention evidence or reasoning. Return one JSON object only with fields answer, answerable, "
            "confidence (0..1), evidence_ids (array), knowledge_used (boolean), missing_information (string).\n"
            f"Original question: {plan.original_question}\n"
            f"Standalone question: {plan.standalone_question}\n"
            f"Question type: {plan.question_type}\n"
            f"Visually selected subject: {subject or plan.visual_target or 'unknown'}\n"
            f"Visible text: {json.dumps(plan.visible_text, ensure_ascii=False)}\n"
            f"Dialogue (assistant text is unverified):\n{format_history(history, self.config.max_history_messages)}\n"
            f"Proposed answer: {candidate.answer}\n"
            f"Evidence: {json.dumps(compact, ensure_ascii=False)}"
        )
        with self._lock:
            self._stats["answer_verification_calls"] += 1
        raw = self._request(image, prompt, self.config.qwen_anchor_model, "answer_verification")
        if not raw:
            return candidate
        answer = " ".join(str(raw.get("answer") or "").split())
        answerable = raw.get("answerable") if isinstance(raw.get("answerable"), bool) else bool(
            answer and "i don't know" not in answer.lower()
        )
        reviewed = AnswerDecision(
            answer=answer or "I don't know.",
            answerable=bool(answerable),
            confidence=clamp01(raw.get("confidence")),
            evidence_ids=[
                str(item).strip()
                for item in (raw.get("evidence_ids", []) or [])
                if str(item).strip()
            ],
            knowledge_used=bool(raw.get("knowledge_used", False)),
            missing_information=str(raw.get("missing_information") or "").strip(),
        )
        if not reviewed.answerable or "i don't know" in reviewed.answer.lower():
            return reviewed if reviewed.confidence >= 0.78 else candidate
        minimum = 0.78 if (not candidate.answerable or "i don't know" in candidate.answer.lower()) else 0.58
        return reviewed if reviewed.confidence >= minimum else candidate

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            values = dict(self._stats)
        calls = (
            values["planner_calls"]
            + values["rerank_calls"]
            + values["audit_calls"]
            + values["answer_verification_calls"]
        )
        values["available"] = self.available
        values["average_latency"] = round(values["latency_total"] / calls, 4) if calls else 0.0
        return values
