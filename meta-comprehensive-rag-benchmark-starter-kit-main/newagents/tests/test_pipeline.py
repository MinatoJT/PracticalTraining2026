from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from newagents import Task1KGAgent, Task2Agent, Task3Agent
from newagents.config import AgentConfig
from newagents.providers.qwen import QwenVisionProvider
from newagents.pipeline import BlackPearlPipeline
from newagents.schemas import AnswerDecision, EvidenceItem, QueryPlan, RerankDecision, extract_json_object
from newagents.tracing import TraceWriter


class FakeSearchPipeline:
    def __init__(self):
        self.image_calls = 0
        self.web_queries = []

    def __call__(self, value, k=10):
        if isinstance(value, Image.Image):
            self.image_calls += 1
            return [{
                "score": 0.91,
                "entities": [{
                    "entity_name": "Eiffel Tower",
                    "entity_attributes": {
                        "location": "Paris, France",
                        "height": "330 metres",
                    },
                }],
            }]
        self.web_queries.append(str(value))
        return [{
            "title": "Eiffel Tower facts",
            "url": "https://example.test/eiffel",
            "snippet": "The Eiffel Tower is in Paris, France. It is about 330 metres tall.",
            "score": 0.82,
        }]


class FakeVision:
    def __init__(self):
        self.plan_histories = []
        self.rerank_calls = 0

    def plan(self, image, question, history, web_enabled):
        self.plan_histories.append(list(history))
        return QueryPlan(
            original_question=question,
            standalone_question=question,
            question_type="location",
            visual_target="Eiffel Tower",
            search_queries=[question, "Eiffel Tower location", "Eiffel Tower facts"] if web_enabled else [],
            requires_web=web_enabled,
            confidence=0.95,
        )

    def rerank(self, image, plan, evidence, history):
        self.rerank_calls += 1
        return RerankDecision(
            answerable=True,
            subject="Eiffel Tower",
            confidence=0.94,
            ranked_scores={item.eid: (0.96 if item.title == "Eiffel Tower" else 0.88) for item in evidence},
        )

    def stats(self):
        return {"fake": True}


class FakeAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, plan, evidence, history, subject, rerank_confidence):
        self.calls.append((plan, evidence, history, subject, rerank_confidence))
        return AnswerDecision(
            answer="It is in Paris, France.",
            answerable=True,
            confidence=0.93,
            evidence_ids=[evidence[0].eid],
        )

    def stats(self):
        return {"fake": True}


class NewAgentsPipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (12, 12), "white")

    @staticmethod
    def _agent(agent_cls, search=None, vision=None, answerer=None):
        task = agent_cls.TASK
        config = AgentConfig.for_task(task)
        config.debug_path = None
        return agent_cls(
            search_pipeline=search or FakeSearchPipeline(),
            config=config,
            vision_provider=vision or FakeVision(),
            answer_provider=answerer or FakeAnswerer(),
        )

    def test_extracts_balanced_json_from_fenced_text(self):
        parsed = extract_json_object('prefix ```json\n{"answer": "Paris", "meta": {"ok": true}}\n``` suffix')
        self.assertEqual(parsed["answer"], "Paris")
        self.assertTrue(parsed["meta"]["ok"])

    def test_task1_never_calls_web_search(self):
        search = FakeSearchPipeline()
        agent = self._agent(Task1KGAgent, search=search)
        answer = agent.batch_generate_response(["Where is this?"], [self.image], [[]])[0]
        self.assertEqual(answer, "It is in Paris, France.")
        self.assertEqual(search.image_calls, 1)
        self.assertEqual(search.web_queries, [])

    def test_task2_uses_multi_query_and_deduplicates_web_evidence(self):
        search = FakeSearchPipeline()
        answerer = FakeAnswerer()
        agent = self._agent(Task2Agent, search=search, answerer=answerer)
        agent.batch_generate_response(["Where is this?"], [self.image], [[]])
        self.assertEqual(len(search.web_queries), 3)
        web_items = [item for item in answerer.calls[0][1] if item.source == "web"]
        self.assertEqual(len(web_items), 1)
        self.assertEqual(len(web_items[0].metadata["queries"]), 3)

    def test_task3_passes_history_and_reuses_image_retrieval(self):
        search = FakeSearchPipeline()
        vision = FakeVision()
        agent = self._agent(Task3Agent, search=search, vision=vision)
        history = [{"role": "user", "content": "What is the landmark?"}, {"role": "assistant", "content": "Eiffel Tower"}]
        agent.batch_generate_response(["Where is it?"], [self.image], [history])
        agent.batch_generate_response(["How tall is it?"], [self.image], [history])
        self.assertEqual(search.image_calls, 1)
        self.assertEqual(vision.plan_histories[0][: len(history)], history)
        self.assertIn("unverified candidate", vision.plan_histories[0][-1]["content"])
        self.assertIn("Eiffel Tower", vision.plan_histories[0][-1]["content"])
        self.assertEqual(agent.visual_pipeline.stats()["image_cache_hits"], 1)

    def test_building_relation_adds_exact_address_query(self):
        plan = QueryPlan(
            original_question="What is the building called where this garage is located?",
            standalone_question="What is the building at 147 Example St called?",
            search_queries=["building at address", "147 Example St"],
        )
        BlackPearlPipeline._refine_relation_queries(plan)
        self.assertEqual(plan.search_queries[-1], '"147 Example St" office tower building name')

    def test_namesake_followup_uses_previous_entity_for_second_hop(self):
        plan = QueryPlan(
            "Where is the person it is named after buried?",
            "Where is the namesake buried?",
            search_queries=["namesake burial"],
        )
        history = [{"role": "assistant", "content": "Example City"}]
        BlackPearlPipeline._refine_namesake_queries(plan, history)
        self.assertEqual(plan.standalone_question, "Where is the person after whom Example City is named buried?")
        self.assertIn('"Example City" named after whom burial place', plan.search_queries)

    def test_namesake_followup_retains_planner_resolved_second_hop(self):
        plan = QueryPlan(
            "Where is the person it is named after buried?",
            "Where is the namesake buried?",
            search_queries=["Example City namesake", "burial place of Person Name"],
        )
        history = [{"role": "assistant", "content": "Example City"}]
        BlackPearlPipeline._refine_namesake_queries(plan, history)
        self.assertEqual(plan.search_queries[-1], "burial place of Person Name")

    def test_high_precision_verifier_skips_simple_confident_identity(self):
        plan = QueryPlan("What brand is this?", "What brand is this?", question_type="identity")
        decision = AnswerDecision("CALLAHEAD", True, 0.95, ["VIS001"])
        self.assertFalse(BlackPearlPipeline._needs_high_precision_verification(plan, decision))

    def test_high_precision_verifier_handles_numeric_and_historical_answers(self):
        numeric = QueryPlan("How much does it cost?", "How much does it cost?", question_type="price")
        historical = QueryPlan(
            "What were the boundary changes?",
            "What were the boundary changes?",
            question_type="historical_geography",
        )
        decision = AnswerDecision("$10", True, 0.95, ["WEB001"])
        self.assertTrue(BlackPearlPipeline._needs_high_precision_verification(numeric, decision))
        self.assertTrue(BlackPearlPipeline._needs_high_precision_verification(historical, decision))

    def test_high_precision_verifier_handles_awards_and_genre_lists(self):
        award = QueryPlan("Has this plant won an award?", "Has this plant won an award?", question_type="factual")
        genres = QueryPlan("Which genres did the author write?", "Which genres?", question_type="author_genre_inquiry")
        decision = AnswerDecision("No", True, 0.95, ["VIS001"])
        self.assertTrue(BlackPearlPipeline._needs_high_precision_verification(award, decision))
        self.assertTrue(BlackPearlPipeline._needs_high_precision_verification(genres, decision))

    def test_task3_reuses_first_turn_anchor_without_repeating_visual_rerank(self):
        vision = FakeVision()
        agent = self._agent(Task3Agent, vision=vision)
        agent.set_trace_contexts([{"session_id": "session-a", "turn_idx": 0}])
        agent.batch_generate_response(["What is this landmark?"], [self.image], [[]])
        history = [
            {"role": "user", "content": "What is this landmark?"},
            {"role": "assistant", "content": "It is the Eiffel Tower."},
        ]
        agent.set_trace_contexts([{"session_id": "session-a", "turn_idx": 1}])
        agent.batch_generate_response(["How tall is it?"], [self.image], [history])
        self.assertEqual(vision.rerank_calls, 1)

    def test_task3_bare_definition_targets_previous_question_concept(self):
        answerer = FakeAnswerer()
        agent = self._agent(Task3Agent, answerer=answerer)
        history = [
            {"role": "user", "content": "How much garden waste does the company recycle each year?"},
            {"role": "assistant", "content": "Several million tonnes."},
        ]
        agent.set_trace_contexts([{"session_id": "session-b", "turn_idx": 1}])
        agent.batch_generate_response(["What is that?"], [self.image], [history])
        self.assertEqual(answerer.calls[0][0].standalone_question, "What is garden waste?")
        self.assertEqual(answerer.calls[0][0].question_type, "definition")

    def test_task3_visual_followup_runs_visual_rerank_again(self):
        vision = FakeVision()
        agent = self._agent(Task3Agent, vision=vision)
        agent.set_trace_contexts([{"session_id": "session-c", "turn_idx": 0}])
        agent.batch_generate_response(["What product is this?"], [self.image], [[]])
        history = [
            {"role": "user", "content": "What product is this?"},
            {"role": "assistant", "content": "A packaged product."},
        ]
        agent.set_trace_contexts([{"session_id": "session-c", "turn_idx": 1}])
        agent.batch_generate_response(["What serving size is visible on this label?"], [self.image], [history])
        self.assertEqual(vision.rerank_calls, 2)

    def test_no_evidence_returns_unknown_without_answer_call(self):
        search = FakeSearchPipeline()
        search.__call__ = lambda value, k=10: []

        class EmptySearch:
            def __call__(self, value, k=10):
                return []

        answerer = FakeAnswerer()
        agent = self._agent(Task1KGAgent, search=EmptySearch(), answerer=answerer)
        answer = agent.batch_generate_response(["What is this?"], [self.image], [[]])[0]
        self.assertEqual(answer, "I don't know.")
        self.assertEqual(answerer.calls, [])

    def test_batch_length_validation(self):
        agent = self._agent(Task1KGAgent)
        with self.assertRaises(ValueError):
            agent.batch_generate_response(["one"], [], [[]])

    def test_missing_api_keys_keep_providers_offline(self):
        with patch.dict("os.environ", {"QWEN_VL_API_KEY": "", "DASHSCOPE_API_KEY": "", "DEEPSEEK_API_KEY": ""}):
            config = AgentConfig.for_task("task1")
            agent = Task1KGAgent(search_pipeline=FakeSearchPipeline(), config=config)
            self.assertFalse(agent.pipeline.vision.available)
            self.assertFalse(agent.pipeline.answerer.available)

    def test_qwen_request_error_does_not_switch_models_automatically(self):
        config = AgentConfig.for_task("task1")
        config.qwen_fallback_model = "different-model"
        provider = QwenVisionProvider(config, TraceWriter(None))
        provider._client = object()
        calls = []

        def fail_request(_image, _prompt, model):
            calls.append(model)
            raise RuntimeError("provider error")

        provider._request_api = fail_request
        result = provider._request(self.image, "{}", "fixed-test-model", "unit_test")
        self.assertEqual(result, {})
        self.assertEqual(calls, ["fixed-test-model"])

    def test_qwen_audits_successful_but_low_confidence_subject(self):
        config = AgentConfig.for_task("task2")
        config.qwen_anchor_model = "anchor-model"
        config.qwen_rerank_model = "rerank-model"
        provider = QwenVisionProvider(config, TraceWriter(None))
        responses = iter([
            {"subject": "Uncertain landmark", "confidence": 0.3, "answerable": True, "ranked": []},
            {"subject": "Eiffel Tower", "confidence": 0.95, "answerable": True, "ranked": []},
        ])
        provider._request = lambda image, prompt, model, purpose: next(responses)
        plan = QueryPlan("Where is it?", "Where is the Eiffel Tower?", visual_target="Eiffel Tower")
        evidence = [EvidenceItem("WEB001", "web", "Eiffel Tower", "Located in Paris")]
        decision = provider.rerank(self.image, plan, evidence, [])
        self.assertEqual(decision.subject, "Eiffel Tower")
        self.assertEqual(provider.stats()["audit_calls"], 1)


if __name__ == "__main__":
    unittest.main()
