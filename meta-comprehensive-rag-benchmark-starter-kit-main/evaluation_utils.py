import re


def _explicit_polarity(text: str):
    """提取明确 Yes/No 极性；没有明确极性时返回 None。"""
    normalized = re.sub(r"[^a-z0-9']+", " ", str(text).lower()).strip()
    negative = r"\b(cannot|can't|could not|is not|was not|does not|did not)\b"
    if re.match(r"^(no|false)\b", normalized) or re.search(negative, normalized):
        return False
    if re.match(r"^(yes|true)\b", normalized):
        return True
    return None


def semantic_shortcut(query: str, ground_truth: str, prediction: str) -> bool:
    """仅对明显语义覆盖做本地捷径；存在极性冲突时必须返回 False。"""
    pred_l = str(prediction).lower()
    gt_l = str(ground_truth).lower()
    if "i don't know" in pred_l or "i don’t know" in pred_l:
        return False

    gt_polarity = _explicit_polarity(gt_l)
    pred_polarity = _explicit_polarity(pred_l)
    if gt_polarity is not None and pred_polarity is not None and gt_polarity != pred_polarity:
        return False

    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    for word, digit in number_words.items():
        pred_l = re.sub(rf"\b{word}\b", digit, pred_l)
        gt_l = re.sub(rf"\b{word}\b", digit, gt_l)

    gt_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", gt_l))
    pred_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", pred_l))
    if gt_numbers and len(gt_numbers & pred_numbers) >= max(1, min(len(gt_numbers), 3)):
        return True

    stop = {
        "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
        "which", "who", "why", "how", "does", "did", "was", "were", "are", "is", "its",
        "into", "about", "there", "their", "have", "has", "had", "can", "could", "would",
        "should", "will", "shall", "than", "then", "them", "they", "you", "your", "because",
        "between", "not", "also", "been", "being", "answer",
    }
    gt_tokens = {token for token in re.findall(r"[a-z0-9]+", gt_l) if len(token) >= 4 and token not in stop}
    pred_tokens = {token for token in re.findall(r"[a-z0-9]+", pred_l) if len(token) >= 4 and token not in stop}
    if not gt_tokens or not pred_tokens:
        return False
    overlap = len(gt_tokens & pred_tokens)
    return overlap >= 3 and overlap / max(1, min(len(gt_tokens), len(pred_tokens))) >= 0.55


def build_deepseek_judge_prompt(query: str, ground_truth: str, prediction: str) -> str:
    """构造干净的中文语义评测 Prompt，要求严格检查实体、数值和 Yes/No 极性。"""
    return (
        "你是问答系统评测器。判断 Prediction 是否正确回答 Question，并覆盖 Ground truth 的关键事实。\n"
        "允许同义改写和更简短的表达，但关键实体、数值、单位、地点和 Yes/No 极性必须一致。\n"
        "如果只答对问题类型却给错主体，或关键词重合但事实相反，必须判为 false。\n"
        "只返回一行 JSON：{\"accuracy\": true} 或 {\"accuracy\": false}。\n\n"
        f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {prediction}\n"
    )
