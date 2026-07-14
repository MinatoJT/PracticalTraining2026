from pathlib import Path


def conversation_interaction_ids(conversation: dict) -> list[str]:
    turns = conversation.get("turns", []) or []
    if isinstance(turns, dict):
        return [str(value) for value in turns.get("interaction_id", []) or []]
    return [str(item.get("interaction_id", "")) for item in turns if isinstance(item, dict)]


def select_fixed_conversations(dataset, sample_ids_file: str):
    """Select conversations by interaction ID while preserving file order."""
    path = Path(sample_ids_file).expanduser().resolve()
    requested_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not requested_ids:
        raise ValueError(f"Fixed sample file is empty: {path}")
    index_by_id = {}
    for index, conversation in enumerate(dataset):
        for interaction_id in conversation_interaction_ids(conversation):
            index_by_id.setdefault(interaction_id, index)
    missing = [interaction_id for interaction_id in requested_ids if interaction_id not in index_by_id]
    if missing:
        raise ValueError(f"Fixed sample IDs not found in dataset: {missing}")
    indices = []
    for interaction_id in requested_ids:
        index = index_by_id[interaction_id]
        if index not in indices:
            indices.append(index)
    return dataset.select(indices), requested_ids, path
