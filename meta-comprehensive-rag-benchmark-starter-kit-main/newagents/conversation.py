from __future__ import annotations

import re
from typing import Any, Dict, List


FOLLOWUP_PATTERN = re.compile(
    r"\b(it|its|this|that|these|those|they|their|he|she|his|her|the company|the city|the building|the animal)\b",
    re.IGNORECASE,
)

BARE_REFERENCE_PATTERN = re.compile(
    r"^\s*(?:what|who)\s+(?:is|are|was|were)\s+(?:this|that|it|these|those|they|them)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def is_unknown(text: str) -> bool:
    normalized = str(text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    return any(
        marker in normalized
        for marker in ("i don't know", "cannot determine", "not enough information", "insufficient information")
    )


def trusted_history(history: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    for message in (history or [])[-limit:]:
        role = str(message.get("role") or "").strip().lower()
        content = " ".join(str(message.get("content") or "").split())[:600]
        if role not in {"user", "assistant"} or not content:
            continue
        if role == "assistant" and is_unknown(content):
            continue
        result.append({"role": role, "content": content})
    return result


def format_history(history: List[Dict[str, Any]], limit: int = 10) -> str:
    messages = trusted_history(history, limit)
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages) or "None"


def fallback_standalone_question(question: str, history: List[Dict[str, Any]], limit: int = 10) -> str:
    current = " ".join(str(question or "").split())
    messages = trusted_history(history, limit)
    if not messages or not FOLLOWUP_PATTERN.search(current):
        return current
    last_user = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
    last_answer = next((item["content"] for item in reversed(messages) if item["role"] == "assistant"), "")
    context = " ".join(value for value in [last_user, last_answer] if value)
    return f"{context} Follow-up question: {current}"[:900]


def recent_question_focus(question: str, history: List[Dict[str, Any]]) -> str:
    """Resolve a bare definition follow-up to the concept requested one turn earlier."""
    if not BARE_REFERENCE_PATTERN.match(str(question or "")):
        return ""
    previous = next(
        (
            " ".join(str(item.get("content") or "").split())
            for item in reversed(history or [])
            if str(item.get("role") or "").strip().lower() == "user"
        ),
        "",
    )
    if not previous:
        return ""
    patterns = (
        r"\bhow\s+(?:much|many)\s+(.+?)\s+(?:does|do|did|is|are|was|were|can|could|would|has|have|had)\b",
        r"\b(?:amount|number|price|weight|height|meaning|definition)\s+of\s+(.+?)(?:[?.!]|$)",
        r"\bwhat\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)(?:[?.!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, previous, flags=re.IGNORECASE)
        if not match:
            continue
        focus = re.sub(r"\b(?:this|that|these|those|the)\b", " ", match.group(1), flags=re.IGNORECASE)
        focus = " ".join(focus.strip(" ,;:-").split())
        if 1 < len(focus) <= 120:
            return focus
    return ""


def infer_question_type(question: str) -> str:
    value = str(question or "").strip().lower()
    if value.startswith(("is ", "are ", "was ", "were ", "can ", "does ", "do ", "did ", "has ", "have ")):
        return "yes_no"
    if "how many" in value or "number of" in value:
        return "count"
    if value.startswith("who"):
        return "person"
    if value.startswith("where"):
        return "location"
    if value.startswith("when") or "what year" in value or "what date" in value:
        return "time"
    if any(term in value for term in ["what is this", "what is shown", "identify", "what kind of", "what type of"]):
        return "identity"
    if any(term in value for term in ["color", "shape", "material", "visible", "look like", "in the image", "in this picture"]):
        return "visual_attribute"
    return "fact"
