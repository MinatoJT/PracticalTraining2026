from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List

from ..config import AgentConfig
from ..conversation import format_history, is_unknown
from ..schemas import AnswerDecision, EvidenceItem, QueryPlan, clamp01, extract_json_object
from ..tracing import TraceWriter


class DeepSeekAnswerProvider:
    """Evidence-constrained text answerer using the existing OpenAI-compatible API."""

    def __init__(self, config: AgentConfig, trace: TraceWriter):
        self.config = config
        self.trace = trace
        self._client = self._build_client()
        self._lock = threading.Lock()
        self._stats = {
            "calls": 0,
            "answer_calls": 0,
            "verification_calls": 0,
            "errors": 0,
            "latency_total": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def _build_client(self):
        key = (os.getenv("DEEPSEEK_API_KEY") or "").strip().strip('"').strip("'")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
        if not key or not base_url.startswith(("http://", "https://")):
            return None
        try:
            from openai import OpenAI

            return OpenAI(api_key=key, base_url=base_url, timeout=self.config.deepseek_timeout, max_retries=0)
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _call(self, messages: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.config.deepseek_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.config.deepseek_max_tokens,
            "response_format": {"type": "json_object"},
        }
        if os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower() != "enabled":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        started = time.perf_counter()
        with self._lock:
            self._stats["calls"] += 1
        try:
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
                raise RuntimeError("deepseek_choices_empty")
            content = str(getattr(choices[0].message, "content", None) or "").strip()
            if not content:
                raise RuntimeError("deepseek_content_empty")
            usage = getattr(response, "usage", None)
            with self._lock:
                self._stats["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                self._stats["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            return content
        except Exception as exc:
            with self._lock:
                self._stats["errors"] += 1
            self.trace.write("deepseek_error", model=self.config.deepseek_model, error=type(exc).__name__)
            return ""
        finally:
            with self._lock:
                self._stats["latency_total"] += time.perf_counter() - started

    def answer(
        self,
        plan: QueryPlan,
        evidence: List[EvidenceItem],
        history: List[Dict[str, Any]],
        subject: str,
        rerank_confidence: float,
    ) -> AnswerDecision:
        if not self.available:
            return self._fallback_answer(plan, evidence, subject)
        with self._lock:
            self._stats["answer_calls"] += 1
        compact = [item.prompt_dict(max_text=900, max_attributes=12) for item in evidence[: self.config.answer_evidence_limit]]
        system = (
            "You are the final answer stage of a multimodal retrieval-augmented QA system. Answer the current question, "
            "not a previous turn. Image-KG evidence is useful for identifying the pictured subject; web evidence is useful "
            "for factual attributes but can be noisy. Never treat an entity-directed web result as proof that the visual "
            "identity is correct. Use supplied evidence first. When visual confidence is reasonable, you may use stable, "
            "widely documented background knowledge, common product/vehicle specifications, general science or safety "
            "knowledge, and basic arithmetic derived from facts in the question. Avoid invented claims and unsupported "
            "changing facts. For a purely binary yes/no question, lead with Yes or No. If the question asks about side "
            "effects, capabilities, reasons, or consequences, include the requested supported specifics after Yes or No. "
            "When the yes/no result depends on a numeric comparison, include the decisive values and one-step calculation. "
            "When sources conflict on a disputed origin or date, state only their supported consensus instead of choosing "
            "a precise claim. Answer only the attribute requested. Do not repeat the visual subject's name or model unless "
            "it is needed to disambiguate the answer; this prevents a correct fact from being spoiled by an uncertain "
            "subtype. For any numerical feasibility question, derive the needed threshold from the values in the question "
            "and use only evidence that matches the stated subject and variant; never substitute a mismatched model. "
            "If the image shows a figure, statue, poster, package, screen, or artwork, a question about the depicted entity "
            "normally refers to the real entity, not the display object's location, unless explicitly stated otherwise. "
            "Earlier assistant messages are unverified context, not evidence. Resolve references from the immediately "
            "preceding dialogue, then verify the resulting entity with current evidence. For list questions, include all "
            "distinct supported items rather than one example. For genre, category, or taxonomy lists, return the shortest "
            "complete set of principal categories explicitly supported by the evidence; do not pad the answer with adjacent "
            "subgenres, themes, demographic labels, or merely plausible classifications. For comparisons, state both sides "
            "and the deciding values "
            "or dates. When a discovery, acquisition, or historical event introduces a person or organization likely to "
            "be referenced next, include that name if evidence supports it. Prefer exact values corroborated by multiple "
            "candidates or an authoritative source. Inspect all supplied evidence before choosing a number or date; the "
            "first or highest-ranked snippet is not automatically correct. When values conflict, prefer a primary source "
            "such as an official manufacturer, publisher, government, university, standards body, or annual report, and "
            "match the exact edition, region, year, trim, submodel, and noun scope in the question. Distinguish a product "
            "family from a named variant, and an earliest patent from a later modern subtype. For 'first', 'latest', or "
            "'most recent' questions, answer the requested scope and time frame rather than a nearby famous event. A bare "
            "definition follow-up normally refers to the main noun or concept requested in the immediately previous user "
            "question, not automatically to the pictured object. Resolve pronouns by semantic type: a question asking "
            "what something 'stands for' needs a name, acronym, or symbol antecedent rather than a nearby age or count. "
            "If a dated yes/no question asks whether a named person held an office and the answer is No, include the "
            "actual office-holder when supported; that entity may be referenced next. After several same-type names are "
            "listed, a singular definite reference normally selects the last-listed compatible name. "
            "For nutrition questions, distinguish per-serving values from the whole package or whole item; if the question "
            "asks what is 'in it', multiply by servings per container when supported. Prefer a supported exact count over "
            "a rounded marketing claim such as 'hundreds' or '300+'. When a cuisine or product type is requested, give "
            "the most concrete supported category instead of a broader label. Confidence must reflect whether the exact "
            "entity, unit basis, and time scope match. Give one value in the convention requested; do not add a competing "
            "edition, launch-year convention, or alternate unit basis unless the question asks for it. "
            "If support is genuinely insufficient, set answerable=false and answer exactly I don't know. Keep an English "
            "answer concise and under 75 tokens. Do not mention retrieval, KG, candidate IDs, web pages, or "
            "reasoning. Return one JSON object only with fields answer, answerable, confidence (0..1), evidence_ids (array), "
            "knowledge_used (boolean), missing_information (string)."
        )
        if self.config.task == "task3" and not history:
            system += (
                " This is the first turn of a multi-turn conversation. When the precise pictured subject, brand, title, "
                "model, or organization is supported, include that name alongside the requested attribute so later "
                "turns retain a searchable entity anchor."
            )
        user = (
            f"Original question: {plan.original_question}\n"
            f"Standalone question: {plan.standalone_question}\n"
            f"Question type: {plan.question_type}\n"
            f"Visually selected subject: {subject or plan.visual_target or 'unknown'}\n"
            f"Visual/rerank confidence: {rerank_confidence:.3f}\n"
            f"Visible image text: {json.dumps(plan.visible_text, ensure_ascii=False)}\n"
            f"Conversation history (assistant text is unverified):\n{format_history(history, self.config.max_history_messages)}\n"
            f"Evidence: {json.dumps(compact, ensure_ascii=False)}"
        )
        raw_text = self._call([{"role": "system", "content": system}, {"role": "user", "content": user}])
        raw = extract_json_object(raw_text)
        if not raw:
            answer = raw_text.strip()
            return AnswerDecision(answer or "I don't know.", bool(answer) and not is_unknown(answer), 0.35 if answer else 0.0, raw=raw_text)
        answer = " ".join(str(raw.get("answer") or "").split())
        answerable = raw.get("answerable") if isinstance(raw.get("answerable"), bool) else bool(answer and not is_unknown(answer))
        evidence_ids = [str(item).strip() for item in (raw.get("evidence_ids", []) or []) if str(item).strip()]
        return AnswerDecision(
            answer=answer or "I don't know.",
            answerable=bool(answerable),
            confidence=clamp01(raw.get("confidence")),
            evidence_ids=evidence_ids,
            knowledge_used=bool(raw.get("knowledge_used", False)),
            missing_information=str(raw.get("missing_information") or "").strip(),
            raw=raw_text,
        )

    def verify(
        self,
        plan: QueryPlan,
        evidence: List[EvidenceItem],
        history: List[Dict[str, Any]],
        subject: str,
        rerank_confidence: float,
        candidate: AnswerDecision,
    ) -> AnswerDecision:
        """Independently audit a Task 2/3 answer against the same retrieved evidence."""
        if not self.available or not candidate.answerable or is_unknown(candidate.answer):
            return candidate
        with self._lock:
            self._stats["verification_calls"] += 1
        compact = [
            item.prompt_dict(max_text=760, max_attributes=10)
            for item in evidence[: self.config.answer_evidence_limit]
        ]
        system = (
            "You are an independent final-answer verifier for a multimodal RAG system. The proposed answer may be "
            "correct, subtly wrong, internally contradictory, or attached to the wrong entity. Resolve the current "
            "question from the dialogue, treating every earlier assistant message as unverified. Check the exact entity, "
            "product variant, edition, region, requested attribute, date scope, numeric unit, serving basis, and comparison "
            "direction. Candidate ranking is not proof: inspect all evidence and prefer directly matching or authoritative "
            "sources. Do not substitute a related model, company, object, historical event, or current value. Preserve the "
            "proposed answer exactly when it is fully supported. When it is clearly wrong but the evidence supports a "
            "correction, return the concise corrected answer. If support is insufficient, return I don't know. For a side-"
            "effect, reason, consequence, or list question, a bare Yes/No is incomplete. For comparisons, do not state one "
            "direction and then contradict it. Return one JSON object only with fields answer, answerable, confidence "
            "(0..1), evidence_ids (array), knowledge_used (boolean), missing_information (string)."
        )
        user = (
            f"Original question: {plan.original_question}\n"
            f"Standalone question: {plan.standalone_question}\n"
            f"Question type: {plan.question_type}\n"
            f"Visually selected subject: {subject or plan.visual_target or 'unknown'}\n"
            f"Visual/rerank confidence: {rerank_confidence:.3f}\n"
            f"Conversation history (assistant text is unverified):\n"
            f"{format_history(history, self.config.max_history_messages)}\n"
            f"Proposed answer: {candidate.answer}\n"
            f"Proposed confidence: {candidate.confidence:.3f}\n"
            f"Proposed evidence IDs: {json.dumps(candidate.evidence_ids, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(compact, ensure_ascii=False)}"
        )
        raw_text = self._call([{"role": "system", "content": system}, {"role": "user", "content": user}])
        raw = extract_json_object(raw_text)
        if not raw:
            return candidate
        answer = " ".join(str(raw.get("answer") or "").split())
        answerable = raw.get("answerable") if isinstance(raw.get("answerable"), bool) else bool(
            answer and not is_unknown(answer)
        )
        confidence = clamp01(raw.get("confidence"))
        reviewed = AnswerDecision(
            answer=answer or "I don't know.",
            answerable=bool(answerable),
            confidence=confidence,
            evidence_ids=[
                str(item).strip()
                for item in (raw.get("evidence_ids", []) or [])
                if str(item).strip()
            ],
            knowledge_used=bool(raw.get("knowledge_used", False)),
            missing_information=str(raw.get("missing_information") or "").strip(),
            raw=raw_text,
        )
        # A weak audit must not erase a stronger grounded answer. A confident
        # refusal, however, is useful for suppressing unsupported hallucinations.
        if not reviewed.answerable or is_unknown(reviewed.answer):
            return reviewed if reviewed.confidence >= 0.78 else candidate
        return reviewed if reviewed.confidence >= 0.48 else candidate

    def _fallback_answer(self, plan: QueryPlan, evidence: List[EvidenceItem], subject: str) -> AnswerDecision:
        if not evidence:
            return AnswerDecision("I don't know.", False, 0.0)
        top = evidence[0]
        title = subject or top.title
        question = plan.standalone_question.lower()
        if plan.question_type == "identity" and title:
            return AnswerDecision(f"This is {title}.", True, min(0.65, top.final_score), [top.eid])
        tokens = {token for token in re.findall(r"[a-z0-9]+", question) if len(token) > 2}
        for item in evidence[:5]:
            for key, value in item.attributes.items():
                key_tokens = set(re.findall(r"[a-z0-9]+", key.lower()))
                if tokens & key_tokens and value:
                    return AnswerDecision(str(value), True, 0.35, [item.eid])
        return AnswerDecision("I don't know.", False, 0.0)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            values = dict(self._stats)
        values["available"] = self.available
        values["average_latency"] = round(values["latency_total"] / values["calls"], 4) if values["calls"] else 0.0
        return values
