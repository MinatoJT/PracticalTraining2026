from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image

from agents.base_agent import BaseAgent

from .config import AgentConfig
from .pipeline import BlackPearlPipeline


class _CRAGMMAgent(BaseAgent):
    TASK = ""

    def __init__(
        self,
        search_pipeline: Any,
        config: Optional[AgentConfig] = None,
        vision_provider: Optional[Any] = None,
        answer_provider: Optional[Any] = None,
    ):
        super().__init__(search_pipeline)
        effective_config = config or AgentConfig.for_task(self.TASK)
        if effective_config.task != self.TASK:
            raise ValueError(f"{type(self).__name__} requires config.task={self.TASK}")
        self.pipeline = BlackPearlPipeline(
            search_pipeline=search_pipeline,
            config=effective_config,
            vision_provider=vision_provider,
            answer_provider=answer_provider,
        )
        self.visual_pipeline = self.pipeline
        self._trace_contexts: List[Dict[str, Any]] = []

    def get_batch_size(self) -> int:
        return 1

    def set_trace_contexts(self, contexts: List[Dict[str, Any]]) -> None:
        self._trace_contexts = [dict(item or {}) for item in (contexts or [])]

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        if not (len(queries) == len(images) == len(message_histories)):
            raise ValueError(
                "Agent batch input lengths differ: "
                f"queries={len(queries)}, images={len(images)}, histories={len(message_histories)}"
            )
        responses: List[str] = []
        for index, (query, image, history) in enumerate(zip(queries, images, message_histories)):
            context = self._trace_contexts[index] if index < len(self._trace_contexts) else {}
            responses.append(self.pipeline.run(query, image, history, context))
        return responses

    def inspect_retrieved_evidence(self, image: Image.Image):
        return self.pipeline.inspect_image_evidence(image)


class Task1KGAgent(_CRAGMMAgent):
    """Task 1: image KG retrieval, visual reranking, evidence-grounded answering."""

    TASK = "task1"


class Task2Agent(_CRAGMMAgent):
    """Task 2: image KG + multi-query web retrieval with cross-modal reranking."""

    TASK = "task2"


class Task3Agent(_CRAGMMAgent):
    """Task 3: Task 2 pipeline plus evidence-checked multi-turn context resolution."""

    TASK = "task3"
