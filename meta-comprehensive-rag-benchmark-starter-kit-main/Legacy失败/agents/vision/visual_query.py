import re
from typing import Any, Dict, Iterable, List


_CATEGORY_TERMS = {
    "car": ("car", "vehicle", "automobile", "sedan", "suv", "truck"),
    "building": ("building", "cathedral", "church", "tower", "museum", "bridge"),
    "food": ("food", "dish", "meal", "fruit", "drink", "tea"),
    "boat": ("boat", "ship", "vessel", "gondola"),
    "person": ("person", "man", "woman", "people"),
    "animal": ("animal", "bird", "fish", "dog", "cat"),
    "product": ("product", "device", "package", "machine", "tool"),
}

_NONVISUAL_PATTERNS = (
    r"\b(?:18|19|20)\d{2}\b",
    r"\b(?:built|build|founded|origin|originated|history|historical|invented)\b",
    r"\b(?:seat|seats|capacity|mpg|mileage|fuel|gallon|wavelength|nanometer|nm)\b",
    r"\b(?:safe|safety|price|cost|nutrition|calorie|caffeine)\b",
    r"\b(?:motor show|exhibition|event|award|released|produced)\b",
)

_VISUAL_CONSTRAINT_PATTERNS = {
    "color": r"\b(?:red|blue|green|yellow|black|white|gray|grey|orange|purple|brown)\b",
    "position": r"\b(?:left|right|top|bottom|center|foreground|background)\b",
    "shape": r"\b(?:round|square|rectangular|long|narrow|tall|short)\b",
    "depiction": r"\b(?:painting|artwork|screen|chart|package|poster|photograph)\b",
}

_VISUAL_ATTRIBUTE_KEYS = {
    "alias", "aliases", "category", "broad_category", "brand", "model",
    "object_type", "type", "color", "colour", "shape", "body_style",
    "visible_logo", "logo", "architectural_style", "style", "material",
    "visible_material", "ocr", "visible_text", "visual_reason", "appearance",
}

_NONVISUAL_ATTRIBUTE_FRAGMENTS = {
    "year", "date", "history", "origin", "capacity", "seat", "mileage", "mpg",
    "fuel", "price", "cost", "nutrition", "calorie", "safe", "safety", "event",
    "exhibition", "built", "construction", "wavelength", "caffeine", "distance",
}


def parse_visual_query_target(query: str, anchor: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Split the visible reference from the knowledge property being requested."""
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    lowered = text.lower()
    anchor = anchor or {}
    target_category = str(anchor.get("generic_category", "")).strip()
    if not target_category:
        for category, terms in _CATEGORY_TERMS.items():
            if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
                target_category = category
                break

    target_reference = str(anchor.get("question_target", "")).strip()
    if not target_reference:
        match = re.search(
            r"\b(this|that|these|those|the)\s+([a-z][a-z -]{1,40}?)(?=\s+(?:in|on|at|with|when|that|which|who|how|what|where|is|was|can|does|did)\b|[?.!,]|$)",
            lowered,
        )
        target_reference = " ".join(match.groups()) if match else (f"the {target_category} shown in the image" if target_category else "the visible object referred to by the user")

    visual_constraints: List[str] = []
    for label, pattern in _VISUAL_CONSTRAINT_PATTERNS.items():
        values = re.findall(pattern, lowered)
        visual_constraints.extend(f"{label}:{value}" for value in values)

    nonvisual_constraints: List[str] = []
    for pattern in _NONVISUAL_PATTERNS:
        nonvisual_constraints.extend(re.findall(pattern, lowered))
    requested_property = text
    return {
        "target_reference": target_reference[:160],
        "target_category": target_category[:80],
        "requested_property": requested_property[:240],
        "visual_constraints": list(dict.fromkeys(visual_constraints)),
        "nonvisual_constraints": list(dict.fromkeys(str(item) for item in nonvisual_constraints)),
    }


def build_visual_candidate_view(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Expose only visually useful candidate fields to the vision reranker."""
    attrs = candidate.get("attributes", {}) or {}
    visual_attrs: Dict[str, str] = {}
    for raw_key, raw_value in attrs.items():
        key = re.sub(r"[^a-z0-9_]+", "_", str(raw_key).strip().lower()).strip("_")
        if not key or any(fragment in key for fragment in _NONVISUAL_ATTRIBUTE_FRAGMENTS):
            continue
        if key in _VISUAL_ATTRIBUTE_KEYS or any(token in key for token in _VISUAL_ATTRIBUTE_KEYS):
            visual_attrs[key] = re.sub(r"\s+", " ", str(raw_value)).strip()[:180]
    return {
        "entity_name": str(candidate.get("entity_name", "")).strip()[:160],
        "visual_attributes": visual_attrs,
        "image_similarity_weak_reference": float(candidate.get("score", 0.0) or 0.0),
        "sources": list(candidate.get("sources", [candidate.get("source", "image_kg")]))[:4],
    }


def normalized_tokens(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower().replace("-", " "))


def token_overlap(left: Any, right: Any) -> float:
    left_tokens, right_tokens = set(normalized_tokens(left)), set(normalized_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
