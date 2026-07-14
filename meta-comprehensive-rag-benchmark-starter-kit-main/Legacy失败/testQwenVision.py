import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from agents.vision.visual_anchor import extract_json_object, validate_anchor, validate_rerank
from agents.vision.qwen_vl_client import QwenVLClient, resolve_model_config
from agents.vision.visual_candidate_pipeline import VisualCandidatePipeline


def _candidate(name, score):
    return {"entity_name": name, "attributes": {"kind": name}, "score": score, "rule_score": score}


class _FakeVisionClient:
    def __init__(self, selected_index=1, fail_anchor=""):
        self.selected_index = selected_index
        self.fail_anchor = fail_anchor
        self.analyze_calls = 0
        self.rerank_calls = 0

    def analyze_image(self, image, query, history):
        self.analyze_calls += 1
        if self.fail_anchor:
            return {"ok": False, "error": self.fail_anchor, "latency": 0.01}
        return {
            "ok": True,
            "error": "",
            "latency": 0.01,
            "anchor": {
                "image_type": "artwork",
                "scene_summary": "A framed artwork depicting a long narrow boat.",
                "question_target": "the boat depicted inside the artwork",
                "primary_subject": "gondola",
                "generic_category": "boat",
                "candidate_entities": [{"name": "gondola", "confidence": 0.92, "visual_reason": "long narrow boat"}],
                "visible_text": [{"text": "VENICE", "location": "caption", "confidence": 0.91}],
                "visual_attributes": {},
                "confidence": 0.92,
                "retrieval_queries": ["gondola long narrow passenger boat", "VENICE boat"],
            },
        }

    def rerank_candidates(self, image, query, anchor, candidates, history):
        self.rerank_calls += 1
        index = min(self.selected_index, len(candidates))
        return {
            "ok": True,
            "error": "",
            "latency": 0.02,
            "rerank": {
                "selected_index": index,
                "selected_entity": candidates[index - 1]["entity_name"],
                "confidence": 0.88,
                "candidate_scores": [
                    {"index": i, "visual_match": 0.95 if i == index else 0.2, "question_match": 0.9 if i == index else 0.2, "ocr_match": 0.1, "final_score": 0.9 if i == index else 0.2, "reason": "visual match"}
                    for i in range(1, len(candidates) + 1)
                ],
                "rejected_indices": [],
                "no_valid_candidate": False,
            },
        }


class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _response(content='{"ok": true}', finish_reason="stop"):
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10, prompt_tokens_details=None)
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="qwen3-vl-flash", usage=usage)


def _client(outcomes, model="qwen3-vl-flash", rerank_model=None, fallback_model="qwen3-vl-flash"):
    with patch.object(QwenVLClient, "_build_client", return_value=None):
        client = QwenVLClient(
            model,
            "https://example.test/v1",
            "TEST_KEY",
            rerank_model=rerank_model,
            fallback_model=fallback_model,
        )
    completions = _FakeCompletions(outcomes)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


class QwenVisionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 24), "white")
        self.legacy = [_candidate("Wrong museum", 0.95), _candidate("Gondola", 0.72)]

    def test_markdown_json_and_ocr_anchor_are_validated(self):
        raw, error = extract_json_object('prefix ```json\n{"primary_subject":"tea chart","confidence":0.9,"visible_text":[]}\n```')
        self.assertEqual(error, "")
        anchor, error = validate_anchor(raw)
        self.assertEqual(error, "")
        self.assertEqual(anchor["confidence"], 0.9)

    def test_out_of_range_confidence_is_rejected(self):
        _, error = validate_anchor({"primary_subject": "tea chart", "confidence": 2.0})
        self.assertEqual(error, "anchor_confidence_out_of_range")

    def test_invalid_json_does_not_create_fake_anchor(self):
        value, error = extract_json_object("not json")
        self.assertEqual(value, {})
        self.assertEqual(error, "invalid_json")

    def test_rerank_index_out_of_range_is_rejected(self):
        _, error = validate_rerank({"selected_index": 9, "no_valid_candidate": False}, 2)
        self.assertEqual(error, "rerank_index_out_of_range")

    def test_vision_disabled_preserves_legacy_candidates(self):
        with patch.dict(os.environ, {"VISION_ENABLED": "0"}, clear=False):
            pipeline = VisualCandidatePipeline()
        result = pipeline.prepare("question", self.image, [], self.legacy)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["candidates"], self.legacy)

    def test_qwen_rerank_becomes_primary_score(self):
        with patch.dict(os.environ, {"VISION_ENABLED": "1", "VISION_RERANK_TOP_N": "15"}, clear=False):
            pipeline = VisualCandidatePipeline()
        pipeline.client = _FakeVisionClient(selected_index=1)
        result = pipeline.prepare("What is the boat?", self.image, [], self.legacy)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["selected_entity"]["entity_name"].lower(), "gondola")
        self.assertEqual(result["candidates"][0]["qwen_final_score"], 0.9)

    def test_service_timeout_falls_back_without_exception(self):
        with patch.dict(os.environ, {"VISION_ENABLED": "1"}, clear=False):
            pipeline = VisualCandidatePipeline()
        pipeline.client = _FakeVisionClient(fail_anchor="vision_timeout")
        result = pipeline.prepare("question", self.image, [], self.legacy)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "vision_timeout")

    def test_cached_anchor_still_reranks_for_new_question(self):
        with patch.dict(os.environ, {"VISION_ENABLED": "1"}, clear=False):
            pipeline = VisualCandidatePipeline()
        fake = _FakeVisionClient()
        pipeline.client = fake
        first = pipeline.prepare("What boat is this?", self.image, [], self.legacy)
        second = pipeline.prepare("What color is it?", self.image, [], first["candidates"], cached_anchor=first["anchor"], refresh_anchor=False)
        self.assertEqual(fake.analyze_calls, 1)
        self.assertEqual(fake.rerank_calls, 2)
        self.assertFalse(second["fallback_used"])

    def test_diagnostic_log_never_contains_base64_or_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vision.jsonl"
            with patch.dict(os.environ, {"VISION_ENABLED": "1", "VISION_DEBUG_PATH": str(path), "QWEN_VL_API_KEY": "SECRET"}, clear=False):
                pipeline = VisualCandidatePipeline()
            pipeline.client = _FakeVisionClient()
            pipeline.prepare("question", self.image, [], self.legacy)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("SECRET", text)
            self.assertNotIn("base64", text)

    def test_dashscope_request_uses_json_mode_and_disables_thinking(self):
        client, completions = _client([_response()])
        result = client._chat_json([{"role": "user", "content": "Return JSON"}], "health", 32)
        self.assertTrue(result["ok"])
        request = completions.calls[0]
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertEqual(request["model"], "qwen3-vl-flash")

    def test_default_models_use_plus_for_anchor_and_rerank(self):
        with patch.dict(os.environ, {}, clear=True):
            anchor, rerank, fallback = resolve_model_config()
        self.assertEqual(anchor, "qwen3.5-omni-plus")
        self.assertEqual(rerank, "qwen3.5-omni-plus")
        self.assertEqual(fallback, "qwen3-vl-flash")

    def test_legacy_model_applies_to_anchor_and_rerank(self):
        with patch.dict(os.environ, {"QWEN_VL_MODEL": "legacy-model"}, clear=True):
            anchor, rerank, _ = resolve_model_config()
        self.assertEqual((anchor, rerank), ("legacy-model", "legacy-model"))

    def test_split_model_variables_are_resolved_independently(self):
        env = {"QWEN_VL_ANCHOR_MODEL": "anchor-model", "QWEN_VL_RERANK_MODEL": "rerank-model"}
        with patch.dict(os.environ, env, clear=True):
            anchor, rerank, _ = resolve_model_config()
        self.assertEqual((anchor, rerank), ("anchor-model", "rerank-model"))

    def test_anchor_and_rerank_methods_use_their_own_models(self):
        client, _ = _client([], model="anchor-model", rerank_model="rerank-model")
        with patch.object(client, "_vision_json", return_value={"ok": False, "error": "test"}) as request:
            client.analyze_image(self.image, "question")
            self.assertEqual(request.call_args.args[3], "anchor-model")
            client.rerank_candidates(self.image, "question", {}, [self.legacy[0]])
            self.assertEqual(request.call_args.args[3], "rerank-model")

    def test_omni_models_do_not_receive_enable_thinking(self):
        for model in ("qwen3.5-omni-plus", "qwen3.5-omni-flash"):
            client, completions = _client([_response()], model=model, fallback_model=model)
            result = client._chat_json([{"role": "user", "content": "JSON"}], "health", 32, model=model)
            self.assertTrue(result["ok"])
            self.assertNotIn("extra_body", completions.calls[0])

    def test_realtime_model_is_rejected_without_api_request(self):
        client, completions = _client([_response()], model="qwen3-omni-flash-realtime")
        result = client._chat_json([], "health", 32, model="qwen3-omni-flash-realtime")
        self.assertEqual(result["error"], "realtime_model_not_supported")
        self.assertEqual(completions.calls, [])

    def test_model_unavailable_falls_back_once(self):
        client, completions = _client(
            [_StatusError(404), _response()],
            model="qwen3.5-omni-plus",
            fallback_model="qwen3-vl-flash",
        )
        result = client._chat_json([], "anchor", 32, model="qwen3.5-omni-plus")
        self.assertTrue(result["ok"])
        self.assertTrue(result["fallback_model_used"])
        self.assertEqual(result["fallback_reason"], "model_not_available")
        self.assertEqual([item["model"] for item in completions.calls], ["qwen3.5-omni-plus", "qwen3-vl-flash"])

    def test_key_and_base_url_are_cleaned(self):
        with patch.object(QwenVLClient, "_build_client", return_value=None):
            client = QwenVLClient("qwen3.5-omni-plus", " https://example.test/v1/// ", "  'SECRET'  ")
        self.assertEqual(client.api_key, "SECRET")
        self.assertEqual(client.base_url, "https://example.test/v1")

    def test_image_request_uses_jpeg_data_url_without_upscaling(self):
        client, completions = _client([_response()])
        client._vision_json(self.image, "请输出 JSON", "anchor", "qwen3-vl-flash")
        content = completions.calls[0]["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[1]["type"], "text")
        self.assertIn("JSON", content[1]["text"])

    def test_qwen_specific_key_has_priority_over_dashscope_key(self):
        QwenVLClient._instances.clear()
        env = {
            "QWEN_VL_API_KEY": "PRIMARY",
            "DASHSCOPE_API_KEY": "SECONDARY",
            "QWEN_VL_BASE_URL": "https://example.test/v1",
        }
        with patch.dict(os.environ, env, clear=False), patch.object(QwenVLClient, "_build_client", return_value=None):
            client = QwenVLClient.shared()
        self.assertEqual(client.api_key, "PRIMARY")
        QwenVLClient._instances.clear()

    def test_401_is_not_retried(self):
        client, completions = _client([_StatusError(401)])
        result = client._chat_json([{"role": "user", "content": "JSON"}], "health", 32)
        self.assertEqual(result["error"], "authentication_error")
        self.assertEqual(len(completions.calls), 1)

    def test_429_is_retried_at_most_twice(self):
        client, completions = _client([_StatusError(429), _StatusError(429), _StatusError(429)])
        with patch("agents.vision.qwen_vl_client.time.sleep"):
            result = client._chat_json([{"role": "user", "content": "JSON"}], "health", 32)
        self.assertEqual(result["error"], "rate_limited")
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(len(completions.calls), 3)

    def test_empty_choices_and_content_are_explicit_failures(self):
        client, _ = _client([SimpleNamespace(choices=[], model="qwen3-vl-flash", usage=None)])
        self.assertEqual(client._chat_json([], "health", 32)["error"], "choices_empty")
        client, _ = _client([_response(content="")])
        self.assertEqual(client._chat_json([], "health", 32)["error"], "content_empty")

    def test_question_specific_anchor_cache_does_not_cross_queries(self):
        with patch.dict(os.environ, {"VISION_ENABLED": "1"}, clear=False):
            pipeline = VisualCandidatePipeline()
        fake = _FakeVisionClient()
        pipeline.client = fake
        pipeline.prepare("What boat is this?", self.image, [], self.legacy)
        pipeline.prepare("What color is it?", self.image, [], self.legacy)
        self.assertEqual(fake.analyze_calls, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
