import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agents.Task1KGAgent import Task1KGAgent
from agents.claim_verifier import apply_claim_verdict, normalize_claim_verdict
from agents.vision.qwen_vl_client import QwenVLClient
from agents.vision.visual_anchor import extract_json_object, normalize_rerank
from agents.vision.visual_candidate_pipeline import VisualCandidatePipeline
from agents.vision.visual_query import build_visual_candidate_view, parse_visual_query_target
from fixed_sample_selection import select_fixed_conversations
from PIL import Image


def candidate(name, score=0.5, **attrs):
    return {
        "entity_name": name, "score": score, "rule_score": score,
        "attributes": attrs, "source": "image_kg", "sources": ["image_kg"],
    }


class FakeDataset(list):
    def select(self, indices):
        return FakeDataset([self[index] for index in indices])


class ControlledRepairTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (24, 24), "white")

    def test_query_parser_separates_nonvisual_property(self):
        target = parse_visual_query_target("Did this red car appear at a 2016 motor show?")
        self.assertEqual(target["target_category"], "car")
        self.assertIn("color:red", target["visual_constraints"])
        self.assertIn("2016", target["nonvisual_constraints"])

    def test_candidate_view_removes_knowledge_attributes(self):
        view = build_visual_candidate_view(candidate(
            "Example sedan", brand="Example", color="red", seating_capacity="five",
            fuel_economy="28 mpg", production_year="2016",
        ))
        self.assertEqual(view["visual_attributes"]["brand"], "Example")
        self.assertEqual(view["visual_attributes"]["color"], "red")
        self.assertNotIn("seating_capacity", view["visual_attributes"])
        self.assertNotIn("fuel_economy", view["visual_attributes"])
        self.assertNotIn("production_year", view["visual_attributes"])

    def test_pure_visual_prompt_forbids_factual_selection(self):
        prompt = QwenVLClient._rerank_prompt(
            "Can this car seat seven people?", {"generic_category": "car"},
            [candidate("Visual match", color="red", seating_capacity="five")], "",
        )
        self.assertIn("not a question-answering model", prompt)
        self.assertIn("must not receive a higher score", prompt)
        self.assertNotIn("seating_capacity", prompt)
        self.assertNotIn("question_match", prompt)

    def test_rerank_has_independent_output_budget(self):
        env = {"QWEN_VL_MAX_TOKENS": "1024", "QWEN_VL_RERANK_MAX_TOKENS": "4096"}
        with patch.dict(os.environ, env, clear=False), patch.object(QwenVLClient, "_build_client", return_value=None):
            client = QwenVLClient("anchor", "https://example.test/v1", "key", rerank_model="rerank")
        client._chat_json = Mock(return_value={"ok": False, "error": "test"})
        client._vision_json(self.image, "prompt", "rerank", "rerank")
        self.assertEqual(client._chat_json.call_args.args[2], 4096)
        client._vision_json(self.image, "prompt", "anchor", "anchor")
        self.assertEqual(client._chat_json.call_args.args[2], 1024)

    def test_complete_new_schema_is_valid(self):
        raw = {
            "selected_index": 1, "selected_entity": "A", "confidence": 0.8,
            "candidate_scores": [{
                "index": 1, "appearance_match": 0.9, "target_reference_match": 0.8,
                "category_match": 1.0, "ocr_match": 0.0, "depiction_level_match": 1.0,
                "scene_consistency": 0.9, "visual_final_score": 0.88,
            }], "no_valid_candidate": False,
        }
        result, error, diagnostics = normalize_rerank(raw, 2)
        self.assertEqual(error, "")
        self.assertEqual(diagnostics["parse_status"], "valid")
        self.assertEqual(result["candidate_scores"][0]["visual_final_score"], 0.88)

    def test_optional_scores_can_be_missing(self):
        result, error, diagnostics = normalize_rerank(
            {"selected_index": 1, "selected_entity": "A", "confidence": 0.8, "no_valid_candidate": False}, 2,
        )
        self.assertEqual(error, "")
        self.assertEqual(result["candidate_scores"], [])

    def test_string_numbers_and_confidence_clamp_are_recovered(self):
        result, error, diagnostics = normalize_rerank(
            {"selected_index": "1", "selected_entity": "A", "confidence": "1.4", "candidate_scores": [], "no_valid_candidate": False}, 2,
        )
        self.assertEqual(error, "")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(diagnostics["parse_status"], "recovered")

    def test_markdown_fence_extracts_first_json(self):
        parsed, error = extract_json_object('text ```json\n{"selected_index":"1"}\n``` trailing')
        self.assertEqual(error, "")
        self.assertEqual(parsed["selected_index"], "1")

    def test_zero_based_indices_are_recovered(self):
        result, error, diagnostics = normalize_rerank(
            {"selected_index": 0, "selected_entity": "A", "candidate_scores": [{"index": 0, "visual_final_score": 0.9}], "no_valid_candidate": False}, 2,
        )
        self.assertEqual(error, "")
        self.assertEqual(result["selected_index"], 1)
        self.assertIn("selected_index_zero_based", diagnostics["recovered_fields"])

    def test_out_of_range_index_remains_fatal(self):
        _, error, _ = normalize_rerank({"selected_index": 8, "no_valid_candidate": False}, 2)
        self.assertEqual(error, "rerank_index_out_of_range")

    def test_selected_entity_mismatch_uses_index(self):
        client = QwenVLClient.__new__(QwenVLClient)
        client.rerank_model = "test"
        payload = '{"selected_index":1,"selected_entity":"wrong","confidence":0.8,"candidate_scores":[],"no_valid_candidate":false}'
        client._vision_json = Mock(return_value={"ok": True, "content": payload})
        result = client.rerank_candidates(self.image, "question", {}, [candidate("Correct")])
        self.assertTrue(result["ok"])
        self.assertEqual(result["rerank"]["selected_entity"], "Correct")
        self.assertEqual(result["rerank_diagnostics"]["parse_status"], "recovered")

    def _pipeline(self):
        with patch.dict(os.environ, {"VISION_ENABLED": "1", "VISION_ANCHOR_FALLBACK_CONFIDENCE": "0.70"}, clear=False):
            pipeline = VisualCandidatePipeline()
        return pipeline

    def test_high_confidence_anchor_survives_rerank_failure(self):
        pipeline = self._pipeline()
        pipeline.client = Mock()
        pipeline.client.anchor_model = "anchor"
        pipeline.client.rerank_model = "rerank"
        pipeline.client.analyze_image.return_value = {
            "ok": True, "anchor": {"primary_subject": "Gondola", "generic_category": "boat", "confidence": 0.95, "candidate_entities": []},
        }
        pipeline.client.rerank_candidates.return_value = {"ok": False, "error": "invalid_json"}
        pipeline.client.stats.return_value = {}
        legacy = [candidate("Wrong museum", 0.95), candidate("Gondola", 0.70)]
        result = pipeline.prepare("What boat is shown?", self.image, [], legacy)
        self.assertEqual(result["selected_entity"]["entity_name"], "Gondola")
        self.assertEqual(result["fallback_level"], "anchor_exact_match")

    def test_generic_anchor_filters_before_legacy_top1(self):
        pipeline = self._pipeline()
        anchor = {"primary_subject": "visible vehicle", "generic_category": "car", "confidence": 0.9, "candidate_entities": []}
        legacy = [candidate("Stone building", 0.95, category="building"), candidate("Red sedan", 0.70, category="car")]
        result = pipeline._hierarchical_fallback(legacy, legacy, anchor, "invalid_json")
        self.assertEqual(result["selected_entity"]["entity_name"], "Red sedan")
        self.assertEqual(result["fallback_level"], "anchor_category_filter")

    def test_no_anchor_uses_legacy_top1(self):
        pipeline = self._pipeline()
        legacy = [candidate("Legacy A", 0.9), candidate("Legacy B", 0.8)]
        result = pipeline._hierarchical_fallback(legacy, legacy, {}, "anchor_failed")
        self.assertEqual(result["selected_entity"]["entity_name"], "Legacy A")
        self.assertEqual(result["fallback_level"], "legacy_image_top1")

    def _retry_agent(self, answer="A complete supported answer."):
        agent = Task1KGAgent.__new__(Task1KGAgent)
        agent.evidence_retry_enabled = True
        agent.evidence_retry_entity_confidence = 0.70
        agent._call_llm = Mock(return_value=answer)
        return agent

    def test_evidence_retry_runs_once_with_direct_evidence(self):
        agent = self._retry_agent()
        selected = candidate("Example sedan", 0.9, seating_capacity="five passengers")
        selected["qwen_selected_confidence"] = 0.9
        answer, metadata = agent._maybe_evidence_retry(
            "How many passengers can this car seat?", "I don't know.", selected,
            [selected], [], [], "retry",
        )
        self.assertEqual(answer, "A complete supported answer.")
        self.assertTrue(metadata["retry_triggered"])
        self.assertEqual(agent._call_llm.call_count, 1)

    def test_retry_does_not_run_for_low_confidence(self):
        agent = self._retry_agent()
        selected = candidate("Example", 0.4, seating_capacity="five")
        answer, metadata = agent._maybe_evidence_retry("How many seats?", "I don't know.", selected, [selected], [], [], "retry")
        self.assertEqual(answer, "I don't know.")
        self.assertFalse(metadata["retry_triggered"])
        agent._call_llm.assert_not_called()

    def test_retry_does_not_run_without_relevant_evidence(self):
        agent = self._retry_agent()
        selected = candidate("Example", 0.9, architect="A person")
        answer, metadata = agent._maybe_evidence_retry("How many seats?", "I don't know.", selected, [selected], [], [], "retry")
        self.assertEqual(metadata["retry_result"], "no_relevant_evidence")
        agent._call_llm.assert_not_called()

    def test_non_idk_never_retries(self):
        agent = self._retry_agent()
        selected = candidate("Example", 0.9, seating_capacity="five")
        answer, metadata = agent._maybe_evidence_retry("How many seats?", "It seats five.", selected, [selected], [], [], "retry")
        self.assertEqual(answer, "It seats five.")
        agent._call_llm.assert_not_called()

    def test_retry_that_remains_idk_stops(self):
        agent = self._retry_agent("I don't know.")
        selected = candidate("Example", 0.9, seating_capacity="five")
        answer, metadata = agent._maybe_evidence_retry("How many seats?", "I don't know.", selected, [selected], [], [], "retry")
        self.assertEqual(answer, "I don't know.")
        self.assertEqual(metadata["retry_result"], "still_idk")
        self.assertEqual(agent._call_llm.call_count, 1)

    def test_claim_verifier_accepts_nonvisual_fact_supported_by_kg(self):
        verdict, error = normalize_claim_verdict({
            "identity_status": "supported", "knowledge_status": "supported",
            "coverage_status": "complete", "decision": "accept",
        })
        self.assertEqual(error, "")
        self.assertEqual(apply_claim_verdict("It seats five.", verdict)[0], "It seats five.")

    def test_claim_verifier_uncertainty_cannot_force_idk(self):
        verdict, error = normalize_claim_verdict({
            "identity_status": "uncertain", "knowledge_status": "insufficient",
            "coverage_status": "complete", "decision": "abstain",
        })
        self.assertEqual(error, "")
        answer, source = apply_claim_verdict("It seats five.", verdict)
        self.assertEqual(answer, "It seats five.")
        self.assertNotEqual(source, "verifier_hard_contradiction")

    def test_claim_verifier_hard_contradiction_can_abstain(self):
        verdict, _ = normalize_claim_verdict({
            "identity_status": "contradicted", "knowledge_status": "insufficient",
            "coverage_status": "off_target", "decision": "abstain",
        })
        self.assertEqual(apply_claim_verdict("Wrong answer.", verdict)[0], "I don't know.")

    def test_fixed_sample_selector_preserves_file_order(self):
        dataset = FakeDataset([
            {"turns": {"interaction_id": ["b"]}},
            {"turns": {"interaction_id": ["a"]}},
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ids.txt"
            path.write_text("a\nb\n", encoding="utf-8")
            selected, ids, _ = select_fixed_conversations(dataset, str(path))
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(selected[0]["turns"]["interaction_id"], ["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
