from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, List

from PIL import Image

from ..config import AgentConfig
from ..schemas import EvidenceItem, clamp01
from ..tracing import TraceWriter


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "; ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ;,\n\t")


def clean_attributes(raw: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in (raw or {}).items():
        name = clean_text(key)
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        content = clean_text(value)
        if name and content:
            result[name[:120]] = content[:1200]
    return result


class ImageKGRetriever:
    def __init__(self, search_pipeline: Any, config: AgentConfig, trace: TraceWriter):
        self.search_pipeline = search_pipeline
        self.config = config
        self.trace = trace

    def retrieve(self, image: Image.Image) -> List[EvidenceItem]:
        if self.search_pipeline is None or not isinstance(image, Image.Image):
            return []
        try:
            rows = self.search_pipeline(image, k=self.config.image_top_k) or []
        except Exception as exc:
            self.trace.write("image_search_error", error=type(exc).__name__)
            return []

        merged: Dict[str, EvidenceItem] = {}
        counter = 0
        total = max(1, len(rows))
        for result_rank, row in enumerate(rows, 1):
            raw_score = float(row.get("score", 0.0) or 0.0)
            rank_score = 1.0 - ((result_rank - 1) / total)
            retrieval_score = max(clamp01(raw_score), 0.35 * rank_score)
            entities = row.get("entities", []) or []
            for entity in entities:
                title = clean_text(entity.get("entity_name") or entity.get("name") or "")
                attributes = clean_attributes(entity.get("entity_attributes") or entity.get("attributes") or {})
                if not title and not attributes:
                    continue
                key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
                if not key:
                    key = json.dumps(attributes, sort_keys=True, ensure_ascii=True)[:240]
                text = "; ".join(f"{name}: {value}" for name, value in attributes.items())
                if key in merged:
                    existing = merged[key]
                    existing.retrieval_score = max(existing.retrieval_score, retrieval_score)
                    existing.attributes.update({k: v for k, v in attributes.items() if k not in existing.attributes})
                    existing.text = "; ".join(f"{name}: {value}" for name, value in existing.attributes.items())
                    continue
                counter += 1
                merged[key] = EvidenceItem(
                    eid=f"IMG{counter:03d}",
                    source="image_kg",
                    title=title or "Unknown image entity",
                    text=text,
                    url=clean_text(row.get("url") or row.get("source_url") or ""),
                    attributes=attributes,
                    retrieval_score=retrieval_score,
                    final_score=retrieval_score,
                    rank=result_rank,
                    metadata={"image_result_rank": result_rank, "raw_score": raw_score},
                )
        result = sorted(merged.values(), key=lambda item: (item.retrieval_score, -item.rank), reverse=True)
        self.trace.write("image_search", results=len(rows), evidence=len(result), top=[item.title for item in result[:5]])
        return result

