from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


@dataclass
class AgentConfig:
    task: str
    web_enabled: bool
    history_enabled: bool
    image_top_k: int = 10
    web_top_k: int = 10
    max_search_queries: int = 3
    web_prefilter_size: int = 14
    rerank_shortlist: int = 14
    answer_evidence_limit: int = 8
    max_chunk_chars: int = 850
    max_answer_words: int = 65
    min_evidence_score: float = 0.08
    min_answer_confidence: float = 0.24
    max_history_messages: int = 10
    qwen_backend: str = "api"
    qwen_anchor_model: str = "qwen3.5-omni-plus-2026-03-15"
    qwen_rerank_model: str = "qwen3-omni-flash-2025-12-01"
    qwen_fallback_model: str = "qwen3-vl-flash"
    qwen_local_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    qwen_timeout: float = 90.0
    qwen_max_tokens: int = 1024
    qwen_max_image_edge: int = 1024
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout: float = 90.0
    deepseek_max_tokens: int = 512
    allow_stable_knowledge: bool = True
    debug_path: Optional[str] = None

    @classmethod
    def for_task(cls, task: str) -> "AgentConfig":
        normalized = str(task).strip().lower()
        if normalized not in {"task1", "task2", "task3"}:
            raise ValueError(f"Unsupported task: {task}")
        web_enabled = normalized in {"task2", "task3"}
        history_enabled = normalized == "task3"
        legacy_model = os.getenv("QWEN_VL_MODEL", "").strip()
        return cls(
            task=normalized,
            web_enabled=web_enabled,
            history_enabled=history_enabled,
            image_top_k=_env_int("NEWAGENTS_IMAGE_TOP_K", 10, 1, 30),
            web_top_k=_env_int("NEWAGENTS_WEB_TOP_K", 30 if web_enabled else 10, 1, 30),
            max_search_queries=_env_int("NEWAGENTS_MAX_SEARCH_QUERIES", 3, 1, 5),
            web_prefilter_size=_env_int("NEWAGENTS_WEB_PREFILTER_SIZE", 24 if web_enabled else 14, 4, 30),
            rerank_shortlist=_env_int("NEWAGENTS_RERANK_SHORTLIST", 24 if web_enabled else 14, 4, 24),
            answer_evidence_limit=_env_int(
                "NEWAGENTS_ANSWER_EVIDENCE",
                14 if normalized == "task3" else (12 if normalized == "task2" else 8),
                3,
                14,
            ),
            max_chunk_chars=_env_int("NEWAGENTS_MAX_CHUNK_CHARS", 850, 300, 1600),
            max_answer_words=_env_int("NEWAGENTS_MAX_ANSWER_WORDS", 65, 20, 75),
            min_evidence_score=_env_float("NEWAGENTS_MIN_EVIDENCE_SCORE", 0.08, 0.0, 1.0),
            min_answer_confidence=_env_float("NEWAGENTS_MIN_ANSWER_CONFIDENCE", 0.24, 0.0, 1.0),
            max_history_messages=_env_int("NEWAGENTS_MAX_HISTORY_MESSAGES", 10, 2, 20),
            qwen_backend=os.getenv("NEWAGENTS_QWEN_BACKEND", "api").strip().lower(),
            qwen_anchor_model=os.getenv(
                "QWEN_VL_ANCHOR_MODEL", legacy_model or "qwen3.5-omni-plus-2026-03-15"
            ).strip(),
            qwen_rerank_model=os.getenv(
                "QWEN_VL_RERANK_MODEL", legacy_model or "qwen3-omni-flash-2025-12-01"
            ).strip(),
            qwen_fallback_model=os.getenv("QWEN_VL_FALLBACK_MODEL", "qwen3-vl-flash").strip(),
            qwen_local_model=os.getenv("NEWAGENTS_QWEN_LOCAL_MODEL", "Qwen/Qwen3-VL-4B-Instruct").strip(),
            qwen_timeout=_env_float("QWEN_VL_TIMEOUT", 90.0, 10.0, 300.0),
            qwen_max_tokens=_env_int("QWEN_VL_MAX_TOKENS", 1024, 128, 4096),
            qwen_max_image_edge=_env_int("QWEN_VL_MAX_IMAGE_EDGE", 1024, 448, 1536),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            deepseek_timeout=_env_float("DEEPSEEK_TIMEOUT", 90.0, 10.0, 300.0),
            deepseek_max_tokens=_env_int("NEWAGENTS_DEEPSEEK_MAX_TOKENS", 512, 128, 2048),
            allow_stable_knowledge=os.getenv("NEWAGENTS_ALLOW_STABLE_KNOWLEDGE", "1") == "1",
            debug_path=os.getenv("NEWAGENTS_DEBUG_PATH"),
        )
