from typing import Any, Dict, List, Tuple


def has_complete_answers(conversation: Dict[str, Any]) -> Tuple[bool, str]:
    """检查多轮会话的每个 turn 是否都有非空 ground truth。"""
    turns = conversation.get("turns", []) or []
    answers = conversation.get("answers", {}) or {}
    if isinstance(answers, dict):
        ids = list(answers.get("interaction_id", []) or [])
        values = list(answers.get("ans_full", []) or [])
        lookup = {str(key): value for key, value in zip(ids, values)}
    else:
        lookup = {
            str(item.get("interaction_id")): item.get("ans_full")
            for item in answers
            if isinstance(item, dict)
        }
    if isinstance(turns, dict):
        turn_ids = list(turns.get("interaction_id", []) or [])
    else:
        turn_ids = [turn.get("interaction_id", "") for turn in turns if isinstance(turn, dict)]
    for raw_interaction_id in turn_ids:
        interaction_id = str(raw_interaction_id)
        if not str(lookup.get(interaction_id, "") or "").strip():
            return False, f"missing_answer:{interaction_id}"
    return True, ""


def valid_conversation_indices(dataset: Any, requested: int) -> Tuple[List[int], List[Dict[str, Any]]]:
    """返回前 requested 个有效会话索引，并记录被排除会话的具体原因。"""
    valid_indices = []
    excluded = []
    for index in range(len(dataset)):
        valid, reason = has_complete_answers(dataset[index])
        if valid:
            valid_indices.append(index)
            if len(valid_indices) >= requested:
                break
        else:
            excluded.append({
                "index": index,
                "session_id": dataset[index].get("session_id"),
                "reason": reason,
            })
    return valid_indices, excluded
