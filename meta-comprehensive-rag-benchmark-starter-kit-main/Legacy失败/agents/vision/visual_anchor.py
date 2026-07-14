import json
import re
from typing import Any, Dict, List, Tuple


def clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _valid_confidence(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def extract_json_object(text: str) -> Tuple[Dict[str, Any], str]:
    """从代码块或少量额外文字中提取第一个完整 JSON 对象。"""
    cleaned = re.sub(r"```(?:json)?", "", str(text or ""), flags=re.I).replace("```", "").strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(value, dict):
                return value, ""
        except json.JSONDecodeError:
            continue
    return {}, "invalid_json"


def validate_anchor(raw: Dict[str, Any], max_candidates: int = 8, max_queries: int = 5) -> Tuple[Dict[str, Any], str]:
    """校验视觉锚点类型；失败时返回明确错误而非虚构空锚点。"""
    if not isinstance(raw, dict):
        return {}, "anchor_not_object"
    for field in ("image_type", "scene_summary", "question_target", "primary_subject", "generic_category"):
        if field in raw and not isinstance(raw[field], str):
            return {}, f"anchor_invalid_{field}"
    if not isinstance(raw.get("candidate_entities", []), list):
        return {}, "anchor_invalid_candidate_entities"
    if not isinstance(raw.get("visible_text", []), list):
        return {}, "anchor_invalid_visible_text"
    if not isinstance(raw.get("retrieval_queries", []), list):
        return {}, "anchor_invalid_retrieval_queries"
    if not isinstance(raw.get("visual_attributes", {}), dict):
        return {}, "anchor_invalid_visual_attributes"
    if "confidence" in raw and not _valid_confidence(raw["confidence"]):
        return {}, "anchor_confidence_out_of_range"
    candidates: List[Dict[str, Any]] = []
    for item in list(raw.get("candidate_entities", []) or [])[:max_candidates]:
        if not isinstance(item, dict):
            continue
        if "confidence" in item and not _valid_confidence(item["confidence"]):
            return {}, "anchor_candidate_confidence_out_of_range"
        name = str(item.get("name", "")).strip()[:160]
        if name:
            candidates.append({"name": name, "confidence": clamp_confidence(item.get("confidence")), "visual_reason": str(item.get("visual_reason", "")).strip()[:300]})
    visible_text = []
    for item in list(raw.get("visible_text", []) or [])[:12]:
        if isinstance(item, dict) and "confidence" in item and not _valid_confidence(item["confidence"]):
            return {}, "anchor_ocr_confidence_out_of_range"
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            visible_text.append({"text": str(item.get("text", "")).strip()[:240], "location": str(item.get("location", "")).strip()[:80], "confidence": clamp_confidence(item.get("confidence"))})
    queries = [str(item).strip()[:160] for item in list(raw.get("retrieval_queries", []) or [])[:max_queries] if str(item).strip()]
    anchor = {
        "image_type": str(raw.get("image_type", "other"))[:40], "scene_summary": str(raw.get("scene_summary", "")).strip()[:500],
        "question_target": str(raw.get("question_target", "")).strip()[:240], "primary_subject": str(raw.get("primary_subject", "")).strip()[:160],
        "generic_category": str(raw.get("generic_category", "")).strip()[:120], "candidate_entities": candidates, "visible_text": visible_text,
        "visual_attributes": raw.get("visual_attributes", {}) if isinstance(raw.get("visual_attributes"), dict) else {},
        "is_depiction_inside_another_object": bool(raw.get("is_depiction_inside_another_object", False)),
        "requires_external_knowledge": bool(raw.get("requires_external_knowledge", False)), "confidence": clamp_confidence(raw.get("confidence")),
        "retrieval_queries": queries,
    }
    if not anchor["primary_subject"] and not candidates:
        return {}, "anchor_missing_subject"
    return anchor, ""


def normalize_rerank(raw: Dict[str, Any], candidate_count: int) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Recover safe rerank fields while keeping fatal index/JSON failures explicit."""
    diagnostics = {"parse_status": "valid", "recovered_fields": [], "warnings": []}
    if not isinstance(raw, dict):
        return {}, "rerank_not_object", diagnostics
    if candidate_count <= 0:
        return {}, "rerank_no_candidates", diagnostics

    raw_no_valid = raw.get("no_valid_candidate", False)
    if isinstance(raw_no_valid, bool):
        no_valid = raw_no_valid
    elif str(raw_no_valid).strip().lower() in {"true", "false"}:
        no_valid = str(raw_no_valid).strip().lower() == "true"
        diagnostics["recovered_fields"].append("no_valid_candidate")
    else:
        return {}, "rerank_invalid_no_valid_candidate", diagnostics

    try:
        selected_index = int(float(raw.get("selected_index", 0)))
        if not isinstance(raw.get("selected_index", 0), int):
            diagnostics["recovered_fields"].append("selected_index")
    except (TypeError, ValueError):
        return {}, "rerank_invalid_index", diagnostics

    raw_scores = raw.get("candidate_scores", [])
    if raw_scores is None:
        raw_scores = []
        diagnostics["recovered_fields"].append("candidate_scores")
    if not isinstance(raw_scores, list):
        return {}, "rerank_invalid_candidate_scores", diagnostics
    score_indices = []
    for item in raw_scores:
        if isinstance(item, dict):
            try:
                score_indices.append(int(float(item.get("index", -1))))
            except (TypeError, ValueError):
                pass
    zero_based = selected_index == 0 and not no_valid and 0 in score_indices
    if zero_based:
        selected_index += 1
        diagnostics["recovered_fields"].append("selected_index_zero_based")
    if not no_valid and not 1 <= selected_index <= candidate_count:
        return {}, "rerank_index_out_of_range", diagnostics

    aliases = {
        "appearance_match": ("appearance_match", "visual_match"),
        "target_reference_match": ("target_reference_match", "question_match"),
        "category_match": ("category_match",), "ocr_match": ("ocr_match",),
        "depiction_level_match": ("depiction_level_match",),
        "scene_consistency": ("scene_consistency",),
        "visual_final_score": ("visual_final_score", "final_score"),
    }
    scores = []
    for raw_item in raw_scores:
        if not isinstance(raw_item, dict):
            diagnostics["warnings"].append("non_object_candidate_score_ignored")
            continue
        try:
            index = int(float(raw_item.get("index", 0))) + (1 if zero_based else 0)
        except (TypeError, ValueError):
            diagnostics["warnings"].append("invalid_candidate_score_index_ignored")
            continue
        if not 1 <= index <= candidate_count:
            diagnostics["warnings"].append("out_of_range_candidate_score_ignored")
            continue
        item = {"index": index}
        for output_field, input_names in aliases.items():
            present = next((name for name in input_names if name in raw_item), None)
            if present is None:
                item[output_field] = None
                continue
            item[output_field] = clamp_confidence(raw_item.get(present))
            if present != output_field or not _valid_confidence(raw_item.get(present)):
                diagnostics["recovered_fields"].append(f"candidate_scores.{output_field}")
        item["reason"] = str(raw_item.get("reason", "")).strip()[:300]
        scores.append(item)

    raw_confidence = raw.get("confidence", 0.0)
    confidence = clamp_confidence(raw_confidence)
    if "confidence" in raw and not _valid_confidence(raw_confidence):
        diagnostics["recovered_fields"].append("confidence_clamped")
    selected_entity = str(raw.get("selected_entity", "")).strip()[:160]
    if not selected_entity and not no_valid:
        diagnostics["warnings"].append("selected_entity_missing_use_index")
    if diagnostics["recovered_fields"] or diagnostics["warnings"]:
        diagnostics["parse_status"] = "recovered"
    return {
        "selected_index": selected_index, "selected_entity": selected_entity,
        "confidence": confidence, "candidate_scores": scores,
        "rejected_indices": raw.get("rejected_indices", []), "no_valid_candidate": no_valid,
    }, "", diagnostics


def validate_rerank(raw: Dict[str, Any], candidate_count: int) -> Tuple[Dict[str, Any], str]:
    rerank, error, _ = normalize_rerank(raw, candidate_count)
    return rerank, error
