from __future__ import annotations

import copy
import hashlib
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import AgentConfig
from .conversation import is_unknown, recent_question_focus
from .providers import DeepSeekAnswerProvider, QwenVisionProvider
from .reasoning import apply_rerank, build_shortlist, evidence_gate
from .retrieval import ImageKGRetriever, WebRetriever
from .schemas import AnswerDecision, EvidenceItem, RerankDecision
from .tracing import TraceWriter


UNKNOWN_ANSWER = "I don't know."


class BlackPearlPipeline:
    """Composition-based evidence-first pipeline shared by all three tasks."""

    def __init__(
        self,
        search_pipeline: Any,
        config: AgentConfig,
        vision_provider: Optional[Any] = None,
        answer_provider: Optional[Any] = None,
    ):
        self.config = config
        self.trace = TraceWriter(config.debug_path)
        self.image_retriever = ImageKGRetriever(search_pipeline, config, self.trace)
        self.web_retriever = WebRetriever(search_pipeline, config, self.trace)
        self.vision = vision_provider or QwenVisionProvider(config, self.trace)
        self.answerer = answer_provider or DeepSeekAnswerProvider(config, self.trace)
        self._image_cache: "OrderedDict[str, List[EvidenceItem]]" = OrderedDict()
        self._conversation_states: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._runs = 0
        self._refusals = 0
        self._cache_hits = 0

    @staticmethod
    def _image_key(image: Image.Image) -> str:
        prepared = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{prepared.width}x{prepared.height}".encode("ascii"))
        digest.update(prepared.tobytes())
        return digest.hexdigest()

    def _image_evidence(self, image: Image.Image) -> List[EvidenceItem]:
        if not isinstance(image, Image.Image):
            return []
        key = self._image_key(image)
        with self._cache_lock:
            cached = self._image_cache.get(key)
            if cached is not None:
                self._image_cache.move_to_end(key)
                self._cache_hits += 1
                return copy.deepcopy(cached)
        items = self.image_retriever.retrieve(image)
        with self._cache_lock:
            self._image_cache[key] = copy.deepcopy(items)
            self._image_cache.move_to_end(key)
            while len(self._image_cache) > 24:
                self._image_cache.popitem(last=False)
        return items

    def inspect_image_evidence(self, image: Image.Image) -> List[EvidenceItem]:
        return self._image_evidence(image)

    @staticmethod
    def _is_visual_question(plan) -> bool:
        question_type = str(plan.question_type or "").lower()
        question = str(plan.original_question or "").lower()
        return any(marker in question_type for marker in ("visual", "ocr", "extract", "identif", "nutrition")) or any(
            marker in question
            for marker in (
                "in the image", "in this picture", "visible", "on the label", "on this label", "this packet",
                "serving", "calorie", "fiber", "protein", "ingredient", "flavor",
            )
        )

    @staticmethod
    def _resolve_bare_reference(plan, history: List[Dict[str, Any]]) -> None:
        focus = recent_question_focus(plan.original_question, history)
        if not focus:
            return
        plan.standalone_question = f"What is {focus}?"
        plan.question_type = "definition"
        plan.visual_target = focus
        plan.search_queries = [plan.standalone_question, f"{focus} definition", f"what does {focus} mean"]
        plan.requires_visual_recheck = False

    @staticmethod
    def _planning_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = list(history or [])
        previous_answer = next(
            (
                " ".join(str(item.get("content") or "").split())
                for item in reversed(result)
                if str(item.get("role") or "").lower() == "assistant" and not is_unknown(item.get("content", ""))
            ),
            "",
        )
        if previous_answer:
            names = []
            for match in re.finditer(r"\b(?:[A-Z][A-Za-z.'’-]*)(?:\s+[A-Z][A-Za-z.'’-]*)*", previous_answer):
                name = " ".join(match.group(0).split()).strip()
                if name.lower() in {"yes", "no", "it", "this", "the", "i"} or len(name) < 3:
                    continue
                if name not in names:
                    names.append(name)
            entity_hint = ""
            if names:
                entity_hint = (
                    " Proper-name candidates in mention order are: " + ", ".join(names[-4:]) + ". A singular definite "
                    "reference normally selects the most recently mentioned type-compatible candidate."
                )
            result.append({
                "role": "user",
                "content": (
                    "Reference-resolution note: the immediately previous assistant answer was "
                    f"'{previous_answer[:400]}'. Treat it as an unverified candidate. Pronouns in the current question "
                    "normally refer to the most recently introduced compatible entity; verify that entity via search. "
                    "Compatibility is semantic: 'stand for' selects a name/acronym/symbol, while quantities and dates "
                    "select entities that can possess that attribute." + entity_hint
                ),
            })
        return result

    @staticmethod
    def _refine_relation_queries(plan) -> None:
        question = str(plan.original_question or "").lower()
        if "building" not in question or not any(term in question for term in ("called", "name", "located", "where")):
            return
        address = re.search(
            r"\b\d{1,6}\s+(?:[A-Za-z0-9.'-]+\s+){0,4}(?:St(?:reet)?|Ave(?:nue)?|Rd|Road|Blvd|Boulevard|Lane|Ln)\b",
            str(plan.standalone_question or ""),
            flags=re.IGNORECASE,
        )
        if not address:
            return
        relation_query = f'"{address.group(0)}" office tower building name'
        existing = [item for item in plan.search_queries if item.lower() != relation_query.lower()]
        plan.search_queries = (existing[:2] + [relation_query])[:3]

    @staticmethod
    def _refine_namesake_queries(plan, history: List[Dict[str, Any]]) -> None:
        question = str(plan.original_question or "")
        lowered = question.lower()
        if "named after" not in lowered and "namesake" not in lowered:
            return
        original_queries = list(plan.search_queries or [])
        previous_answer = next(
            (
                " ".join(str(item.get("content") or "").split()).strip(" .")
                for item in reversed(history or [])
                if str(item.get("role") or "").lower() == "assistant"
                and not is_unknown(str(item.get("content") or ""))
            ),
            "",
        )
        if not previous_answer or len(previous_answer.split()) > 12:
            return
        if "buried" in lowered or "burial" in lowered:
            plan.standalone_question = f"Where is the person after whom {previous_answer} is named buried?"
            relation_queries = [
                f'"{previous_answer}" named after whom burial place',
                f'"{previous_answer}" namesake biography buried',
            ]
        else:
            plan.standalone_question = f"Who or what is {previous_answer} named after?"
            relation_queries = [
                f'"{previous_answer}" named after whom',
                f'"{previous_answer}" origin of name namesake',
            ]
        plan.visual_target = previous_answer
        # Keep one planner-produced query: it may already contain the resolved
        # intermediate person's exact name, which is essential for hop two.
        exact_original = next(
            (
                query
                for query in reversed(original_queries)
                if any(marker in query.lower() for marker in ("buried", "burial", "grave", "tomb"))
                and previous_answer.lower() not in query.lower()
            ),
            "",
        )
        candidates = [plan.standalone_question, relation_queries[0], exact_original or relation_queries[1]]
        plan.search_queries = []
        for query in candidates:
            if query and query.lower() not in {item.lower() for item in plan.search_queries}:
                plan.search_queries.append(query)
        plan.requires_visual_recheck = False

    @staticmethod
    def _needs_high_precision_verification(plan, decision: AnswerDecision) -> bool:
        """Route only high-risk factual answers to the expensive final adjudicator."""
        if not decision.answerable or is_unknown(decision.answer) or decision.confidence < 0.78:
            return True
        question_type = str(plan.question_type or "").lower()
        question = str(plan.original_question or "").lower()
        type_markers = (
            "histor", "timeline", "statistic", "numeric", "price", "cost", "comparison", "comparative",
            "temporal", "date", "count", "quantity", "list", "nutrition", "product_attribute", "genre",
        )
        text_markers = (
            "how many", "how much", "what percent", "percentage", "price", "cost", "msrp", "more than",
            "less than", "compare", "which has more", "which is more", "between", "from ", "buried",
            "boundary", "boundaries", "side effect", "consequence", "first ", "latest ", "most recent",
            "award", "substitute", "replace", "genres",
        )
        return any(marker in question_type for marker in type_markers) or any(
            marker in question for marker in text_markers
        )

    def run(
        self,
        question: str,
        image: Image.Image,
        history: List[Dict[str, Any]],
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._runs += 1
        context = dict(trace_context or {})
        history = list(history or []) if self.config.history_enabled else []
        session_id = str(context.get("session_id") or "")
        turn_idx = int(context.get("turn_idx", 0) or 0)
        state: Dict[str, Any] = {}
        if session_id:
            with self._cache_lock:
                state = dict(self._conversation_states.get(session_id, {}))
        try:
            planning_history = self._planning_history(history) if self.config.history_enabled else history
            current_question = str(question or "").strip()
            plan = self.vision.plan(image, current_question, planning_history, self.config.web_enabled)
            if self.config.task == "task3":
                self._resolve_bare_reference(plan, history)
                self._refine_namesake_queries(plan, history)
                self._refine_relation_queries(plan)
            image_items = self._image_evidence(image)
            web_items = self.web_retriever.retrieve(plan) if self.config.web_enabled else []
            shortlist = build_shortlist(image_items, web_items, plan, self.config)
            reuse_visual_anchor = (
                self.config.task == "task3"
                and turn_idx > 0
                and state.get("visual_anchor")
                and not self._is_visual_question(plan)
            )
            if reuse_visual_anchor:
                rerank = RerankDecision(
                    answerable=True,
                    subject=plan.visual_target or str(state["visual_anchor"]),
                    confidence=max(0.7, plan.confidence),
                    reason="reuse_conversation_visual_anchor",
                )
            else:
                rerank = self.vision.rerank(image, plan, shortlist, planning_history)
            ranked = apply_rerank(shortlist, rerank, plan)
            answerable, gate_reason = evidence_gate(ranked, rerank, self.config)
            self.trace.write(
                "evidence_gate",
                **context,
                task=self.config.task,
                question=question,
                image_evidence=len(image_items),
                web_evidence=len(web_items),
                shortlist=len(ranked),
                top_score=round(ranked[0].final_score, 4) if ranked else 0.0,
                planned_subject=plan.visual_target,
                reranked_subject=rerank.subject,
                rerank_confidence=rerank.confidence,
                answerable=answerable,
                reason=gate_reason,
            )
            if not answerable:
                self._refusals += 1
                return UNKNOWN_ANSWER

            visual_subject = rerank.subject or plan.visual_target or str(state.get("visual_anchor") or "")
            visual_confidence = plan.confidence if rerank.reason == "provider_fallback" else rerank.confidence
            answer_evidence = ranked
            if visual_subject and visual_confidence >= 0.65 and not reuse_visual_anchor:
                visual_text = "; ".join(rerank.visual_facts)
                visual = EvidenceItem(
                    eid="VIS001",
                    source="visual_observation",
                    title=visual_subject,
                    text=visual_text or f"Visually identified subject: {visual_subject}",
                    retrieval_score=visual_confidence,
                    rerank_score=visual_confidence,
                    final_score=visual_confidence,
                    metadata={"direct_visual_observation": True},
                )
                answer_evidence = [visual] + ranked

            decision = self.answerer.answer(
                plan=plan,
                evidence=answer_evidence,
                history=history,
                subject=visual_subject,
                rerank_confidence=visual_confidence,
            )
            needs_high_precision = self._needs_high_precision_verification(plan, decision)
            if (
                hasattr(self.answerer, "verify")
                and (self.config.task in {"task2", "task3"} or needs_high_precision)
            ):
                decision = self.answerer.verify(
                    plan=plan,
                    evidence=answer_evidence,
                    history=history,
                    subject=visual_subject,
                    rerank_confidence=visual_confidence,
                    candidate=decision,
                )
            needs_high_precision = self._needs_high_precision_verification(plan, decision)
            if hasattr(self.vision, "verify_answer") and needs_high_precision:
                decision = self.vision.verify_answer(
                    image=image,
                    plan=plan,
                    evidence=answer_evidence,
                    history=history,
                    subject=visual_subject,
                    candidate=decision,
                )
            answer = self._finalize(decision, answer_evidence)
            if session_id:
                next_state = dict(state)
                if turn_idx == 0 and visual_subject and visual_confidence >= 0.7:
                    next_state["visual_anchor"] = visual_subject
                    next_state["visual_confidence"] = visual_confidence
                next_state["last_answer"] = answer
                with self._cache_lock:
                    self._conversation_states[session_id] = next_state
                    self._conversation_states.move_to_end(session_id)
                    while len(self._conversation_states) > 128:
                        self._conversation_states.popitem(last=False)
            if is_unknown(answer):
                self._refusals += 1
            self.trace.write(
                "answer",
                **context,
                task=self.config.task,
                question=question,
                answer=answer,
                confidence=decision.confidence,
                evidence_ids=decision.evidence_ids,
                knowledge_used=decision.knowledge_used,
            )
            return answer
        except Exception as exc:
            self._refusals += 1
            self.trace.write("pipeline_error", **context, task=self.config.task, error=type(exc).__name__)
            return UNKNOWN_ANSWER

    def _finalize(self, decision: AnswerDecision, evidence: List[EvidenceItem]) -> str:
        answer = " ".join(str(decision.answer or "").split())
        if not decision.answerable or not answer or is_unknown(answer):
            return UNKNOWN_ANSWER
        if decision.confidence < self.config.min_answer_confidence:
            return UNKNOWN_ANSWER

        valid_ids = {item.eid for item in evidence[: self.config.answer_evidence_limit]}
        decision.evidence_ids = [eid for eid in decision.evidence_ids if eid in valid_ids]
        if not decision.evidence_ids and not (decision.knowledge_used and self.config.allow_stable_knowledge):
            return UNKNOWN_ANSWER

        answer = re.sub(r"^(answer\s*:\s*)", "", answer, flags=re.IGNORECASE).strip()
        words = answer.split()
        if len(words) > self.config.max_answer_words:
            answer = " ".join(words[: self.config.max_answer_words]).rstrip(" ,;:") + "."
        return answer or UNKNOWN_ANSWER

    def stats(self) -> Dict[str, Any]:
        return {
            "task": self.config.task,
            "runs": self._runs,
            "refusals": self._refusals,
            "image_cache_hits": self._cache_hits,
            "image_cache_size": len(self._image_cache),
            "qwen": self.vision.stats() if hasattr(self.vision, "stats") else {},
            "deepseek": self.answerer.stats() if hasattr(self.answerer, "stats") else {},
        }
