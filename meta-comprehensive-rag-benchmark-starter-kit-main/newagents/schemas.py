from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse the first balanced JSON object from a model response."""
    raw = str(text or "").strip()
    if not raw:
        return {}
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(raw[start : index + 1])
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


@dataclass
class QueryPlan:
    original_question: str
    standalone_question: str
    question_type: str = "fact"
    visual_target: str = ""
    visible_text: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    requires_web: bool = False
    requires_visual_recheck: bool = True
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], original_question: str, web_enabled: bool) -> "QueryPlan":
        standalone = str(raw.get("standalone_question") or original_question).strip()
        queries = []
        for value in raw.get("search_queries", []) or []:
            value = " ".join(str(value).split())
            if value and value.lower() not in {item.lower() for item in queries}:
                queries.append(value[:300])
        if web_enabled and standalone and standalone.lower() not in {item.lower() for item in queries}:
            queries.insert(0, standalone[:300])
        return cls(
            original_question=original_question,
            standalone_question=standalone or original_question,
            question_type=str(raw.get("question_type") or "fact").strip().lower(),
            visual_target=str(raw.get("visual_target") or "").strip(),
            visible_text=[str(item).strip() for item in (raw.get("visible_text", []) or []) if str(item).strip()][:10],
            search_queries=queries,
            requires_web=bool(raw.get("requires_web", web_enabled)) and web_enabled,
            requires_visual_recheck=bool(raw.get("requires_visual_recheck", True)),
            confidence=clamp01(raw.get("confidence")),
        )


@dataclass
class EvidenceItem:
    eid: str
    source: str
    title: str
    text: str
    url: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    retrieval_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    query_id: str = ""
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prompt_dict(self, max_text: int = 900, max_attributes: int = 12) -> Dict[str, Any]:
        attrs = dict(list(self.attributes.items())[:max_attributes])
        return {
            "id": self.eid,
            "source": self.source,
            "title": self.title,
            "text": self.text[:max_text],
            "attributes": attrs,
            "url": self.url,
            "retrieval_score": round(self.retrieval_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "final_score": round(self.final_score, 4),
        }


@dataclass
class RerankDecision:
    answerable: Optional[bool] = None
    subject: str = ""
    confidence: float = 0.0
    ranked_scores: Dict[str, float] = field(default_factory=dict)
    visual_facts: List[str] = field(default_factory=list)
    missing_information: str = ""
    reason: str = ""


@dataclass
class AnswerDecision:
    answer: str
    answerable: bool
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    knowledge_used: bool = False
    missing_information: str = ""
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
