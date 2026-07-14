from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set, Tuple

from ..config import AgentConfig
from ..schemas import EvidenceItem, QueryPlan, clamp01
from ..tracing import TraceWriter
from .image_kg import clean_text


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its", "from", "by", "as", "at", "what",
    "which", "who", "where", "when", "why", "how", "does", "do", "did", "can", "could", "would", "should",
    "image", "picture", "shown", "showing", "provide", "answer", "question",
}


def tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]*", str(text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def lexical_relevance(query: str, text: str) -> float:
    query_tokens = tokens(query)
    text_tokens = tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    coverage = overlap / len(query_tokens)
    precision = overlap / min(len(text_tokens), 40)
    phrase_bonus = 0.12 if len(query.strip()) >= 5 and query.lower() in text.lower() else 0.0
    return clamp01(0.72 * coverage + 0.28 * precision + phrase_bonus)


def split_chunks(text: str, max_chars: int) -> List[str]:
    compact = " ".join(str(text or "").split())
    if not compact:
        return []
    if len(compact) <= max_chars:
        return [compact]
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            while words:
                part: List[str] = []
                size = 0
                while words and size + len(words[0]) + 1 <= max_chars:
                    word = words.pop(0)
                    part.append(word)
                    size += len(word) + 1
                if part:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(" ".join(part))
                else:
                    chunks.append(words.pop(0)[:max_chars])
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class WebRetriever:
    def __init__(self, search_pipeline: Any, config: AgentConfig, trace: TraceWriter):
        self.search_pipeline = search_pipeline
        self.config = config
        self.trace = trace

    @staticmethod
    def _fields(row: Dict[str, Any]) -> Tuple[str, str, str, float]:
        title = clean_text(row.get("page_name") or row.get("title") or row.get("name") or "")
        url = clean_text(row.get("page_url") or row.get("url") or row.get("source_url") or "")
        snippet = clean_text(
            row.get("page_snippet")
            or row.get("snippet")
            or row.get("description")
            or row.get("summary")
            or row.get("text")
            or row.get("content")
            or row.get("page_content")
            or ""
        )
        score = float(row.get("score", 0.0) or 0.0)
        return title, url, snippet, score

    def retrieve(self, plan: QueryPlan) -> List[EvidenceItem]:
        if self.search_pipeline is None or not self.config.web_enabled:
            return []
        queries = list(plan.search_queries[: self.config.max_search_queries])
        if plan.standalone_question and plan.standalone_question.lower() not in {item.lower() for item in queries}:
            queries.insert(0, plan.standalone_question)
        queries = queries[: self.config.max_search_queries]

        merged: Dict[str, EvidenceItem] = {}
        counter = 0
        for query_index, query in enumerate(queries, 1):
            try:
                rows = self.search_pipeline(query, k=self.config.web_top_k) or []
            except Exception as exc:
                self.trace.write("web_search_error", query=query, error=type(exc).__name__)
                continue
            for result_rank, row in enumerate(rows, 1):
                title, url, snippet, raw_score = self._fields(row)
                if not title and not snippet:
                    continue
                chunks = split_chunks(snippet, self.config.max_chunk_chars) or [title]
                for chunk_index, chunk in enumerate(chunks, 1):
                    combined = f"{title} {chunk}"
                    lexical = max(
                        lexical_relevance(plan.standalone_question, combined),
                        lexical_relevance(query, combined),
                    )
                    rrf = 1.0 / (20.0 + result_rank)
                    retrieval = max(clamp01(raw_score), min(1.0, 5.0 * rrf))
                    key = (url.lower() if url else title.lower()) + "|" + re.sub(r"\W+", " ", chunk.lower())[:180]
                    if key in merged:
                        existing = merged[key]
                        existing.lexical_score = max(existing.lexical_score, lexical)
                        existing.retrieval_score = max(existing.retrieval_score, retrieval)
                        existing.metadata.setdefault("queries", []).append(query)
                        query_ids = existing.metadata.setdefault("query_ids", [])
                        if f"Q{query_index}" not in query_ids:
                            query_ids.append(f"Q{query_index}")
                        continue
                    counter += 1
                    merged[key] = EvidenceItem(
                        eid=f"WEB{counter:03d}",
                        source="web",
                        title=title or "Web result",
                        text=chunk,
                        url=url,
                        retrieval_score=retrieval,
                        lexical_score=lexical,
                        final_score=0.55 * lexical + 0.45 * retrieval,
                        query_id=f"Q{query_index}",
                        rank=result_rank,
                        metadata={
                            "queries": [query],
                            "query_ids": [f"Q{query_index}"],
                            "chunk_index": chunk_index,
                            "raw_score": raw_score,
                        },
                    )
        ranked = sorted(merged.values(), key=lambda item: (item.final_score, item.lexical_score), reverse=True)
        # Preserve query diversity before filling by global score. A generic first
        # query must not crowd out the exact-entity or official-source query.
        result: List[EvidenceItem] = []
        source_counts: Dict[str, int] = {}

        def add(item: EvidenceItem) -> bool:
            if item.eid in {chosen.eid for chosen in result}:
                return False
            source_key = (item.url or item.title).strip().lower()
            count = source_counts.get(source_key, 0)
            if count >= 2:
                return False
            source_counts[source_key] = count + 1
            result.append(item)
            return True

        per_query = max(2, self.config.web_prefilter_size // (len(queries) + 1))
        for query_index in range(1, len(queries) + 1):
            taken = 0
            query_id = f"Q{query_index}"
            for item in ranked:
                if query_id not in item.metadata.get("query_ids", [item.query_id]):
                    continue
                if add(item):
                    taken += 1
                if taken >= per_query or len(result) >= self.config.web_prefilter_size:
                    break
            if len(result) >= self.config.web_prefilter_size:
                break

        for item in ranked:
            add(item)
            if len(result) >= self.config.web_prefilter_size:
                break
        result.sort(key=lambda item: (item.final_score, item.lexical_score), reverse=True)
        self.trace.write("web_search", queries=queries, evidence=len(result), top=[item.title for item in result[:5]])
        return result
