from __future__ import annotations

from typing import List, Tuple

from ..config import AgentConfig
from ..retrieval.web import lexical_relevance
from ..schemas import EvidenceItem, QueryPlan, RerankDecision, clamp01


def _base_score(item: EvidenceItem, plan: QueryPlan) -> float:
    searchable = f"{item.title} {item.text} " + " ".join(f"{key} {value}" for key, value in item.attributes.items())
    item.lexical_score = max(item.lexical_score, lexical_relevance(plan.standalone_question, searchable))
    source_bonus = 0.0
    if plan.question_type in {"identity", "visual_attribute", "count"} and item.source == "image_kg":
        source_bonus = 0.08
    elif plan.question_type not in {"identity", "visual_attribute"} and item.source == "web":
        source_bonus = 0.04
    return clamp01(0.52 * item.retrieval_score + 0.48 * item.lexical_score + source_bonus)


def build_shortlist(image_items: List[EvidenceItem], web_items: List[EvidenceItem], plan: QueryPlan, config: AgentConfig) -> List[EvidenceItem]:
    for item in image_items + web_items:
        item.final_score = _base_score(item, plan)
    ranked_image = sorted(image_items, key=lambda item: item.final_score, reverse=True)
    ranked_web = sorted(web_items, key=lambda item: item.final_score, reverse=True)
    if not config.web_enabled:
        return ranked_image[: config.rerank_shortlist]

    visual_types = {"identity", "visual_attribute", "count", "object_identification", "entity_identification", "extraction"}
    is_visual_question = plan.question_type in visual_types or any(
        marker in plan.question_type for marker in ("identif", "visual", "ocr", "extract")
    )
    if config.task in {"task2", "task3"} and not is_visual_question:
        image_quota = min(len(ranked_image), 3)
    else:
        image_quota = min(len(ranked_image), max(4, config.rerank_shortlist // 2))
    web_quota = min(len(ranked_web), config.rerank_shortlist - image_quota)
    selected = ranked_image[:image_quota] + ranked_web[:web_quota]
    if len(selected) < config.rerank_shortlist:
        selected_ids = {item.eid for item in selected}
        remainder = [item for item in ranked_image + ranked_web if item.eid not in selected_ids]
        selected.extend(remainder[: config.rerank_shortlist - len(selected)])
    return sorted(selected, key=lambda item: item.final_score, reverse=True)


def apply_rerank(items: List[EvidenceItem], decision: RerankDecision, plan: QueryPlan) -> List[EvidenceItem]:
    has_model_scores = bool(decision.ranked_scores)
    for item in items:
        item.rerank_score = decision.ranked_scores.get(item.eid, 0.0)
        base = _base_score(item, plan)
        if has_model_scores:
            item.final_score = clamp01(0.64 * item.rerank_score + 0.20 * item.lexical_score + 0.16 * item.retrieval_score)
        else:
            item.final_score = base
    return sorted(items, key=lambda item: (item.final_score, item.rerank_score, item.retrieval_score), reverse=True)


def evidence_gate(items: List[EvidenceItem], decision: RerankDecision, config: AgentConfig) -> Tuple[bool, str]:
    if not items:
        return False, "no_evidence"
    top_score = items[0].final_score
    if top_score < config.min_evidence_score:
        return False, "low_evidence_score"
    # A negative visual decision blocks answering only when no usable subject was found.
    # Factual support is handled by the final evidence-constrained answerer.
    if decision.answerable is False and decision.confidence >= 0.82 and not decision.subject:
        return False, "confident_unanswerable"
    return True, "evidence_available"
