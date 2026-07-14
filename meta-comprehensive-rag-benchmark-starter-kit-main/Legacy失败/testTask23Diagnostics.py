import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agents.Task1KGAgent import Task1KGAgent
from agents.Task2Agent import Task2Agent
from agents.Task3Agent import Task3Agent
from conversation_validation import has_complete_answers, valid_conversation_indices
from evaluation_utils import build_deepseek_judge_prompt, semantic_shortcut


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _response(content="", reasoning="", finish="stop", choices=True):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        refusal=None,
        tool_calls=None,
    )
    choice_list = [SimpleNamespace(message=message, finish_reason=finish)] if choices else []
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13)
    return SimpleNamespace(choices=choice_list, usage=usage, model="deepseek-v4-flash")


class Task23DiagnosticsTests(unittest.TestCase):
    def _task1_for_llm(self, response):
        agent = Task1KGAgent.__new__(Task1KGAgent)
        completions = _FakeCompletions(response)
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        agent.model_name = "deepseek-v4-flash"
        agent.deepseek_thinking = "disabled"
        agent.debug_path = None
        return agent, completions

    def test_reasoning_only_is_not_used_as_final_answer(self):
        agent, completions = self._task1_for_llm(_response(reasoning="private reasoning", finish="length"))
        answer = agent._call_llm([{"role": "user", "content": "test"}], 8, "test")
        self.assertEqual(answer, "")
        self.assertEqual(completions.kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_empty_choices_returns_empty_content(self):
        agent, _ = self._task1_for_llm(_response(choices=False))
        self.assertEqual(agent._call_llm([], 8, "test"), "")

    def test_entity_directed_web_does_not_vote_for_kg(self):
        agent = Task2Agent.__new__(Task2Agent)
        candidates = [
            {"entity_name": "Correct A", "rule_score": 0.60},
            {"entity_name": "Wrong B", "rule_score": 0.58},
        ]
        broad = [{"title": "Correct A facts", "snippet": "Correct A details"}]
        ranked = agent._rerank_kg_with_web(candidates, broad)
        self.assertEqual(ranked[0]["entity_name"], "Correct A")

    def test_visual_anchor_is_added_to_context(self):
        agent = Task3Agent.__new__(Task3Agent)
        agent.context_turn_limit = 8
        anchor = {"selected_entity": {"entity_name": "Gondola"}, "candidates": []}
        context = agent._build_context_state([], anchor)
        self.assertEqual(context["recent_entities"][0], "Gondola")

    def test_new_visual_evidence_can_correct_anchor(self):
        agent = Task3Agent.__new__(Task3Agent)
        agent.rerank_top_n = 3
        agent.visual_anchors = {}
        agent._update_visual_anchor("s1", {"entity_name": "Wrong"}, [{"entity_name": "Wrong", "score": 0.7}], 0)
        agent._update_visual_anchor("s1", {"entity_name": "Correct"}, [{"entity_name": "Correct", "score": 0.9}], 1)
        self.assertEqual(agent.visual_anchors["s1"]["selected_entity"]["entity_name"], "Correct")

    def test_image_score_remains_primary_over_small_category_bonus(self):
        agent = Task3Agent.__new__(Task3Agent)
        correct = {"entity_name": "Museum", "attributes": {}, "score": 0.80}
        category_match = {"entity_name": "Gondola", "attributes": {}, "score": 0.70}
        self.assertGreater(
            agent._score_candidate_by_rules("what boat is shown", correct),
            agent._score_candidate_by_rules("what boat is shown", category_match),
        )

    def test_complete_short_question_is_not_forced_followup(self):
        agent = Task3Agent.__new__(Task3Agent)
        self.assertFalse(agent._looks_like_followup("Who built this cathedral?".replace("this ", "the ")))
        self.assertTrue(agent._looks_like_followup("What color is it?"))

    def test_correct_short_answer_is_not_overwritten(self):
        agent = Task3Agent.__new__(Task3Agent)
        self.assertFalse(agent._needs_sentence_rewrite("It is Venice.", "Where is it?", []))

    def test_complete_sentence_with_entity_and_new_verb_is_kept(self):
        agent = Task3Agent.__new__(Task3Agent)
        candidates = [{"entity_name": "Hypericum punctatum"}]
        answer = "Droopy leaves on Hypericum punctatum typically indicate overwatering."
        self.assertFalse(agent._needs_sentence_rewrite(answer, "Why are its leaves droopy?", candidates))

    def test_batch_length_mismatch_is_explicit(self):
        agent = Task2Agent.__new__(Task2Agent)
        with self.assertRaisesRegex(ValueError, "长度不一致"):
            agent.batch_generate_response(["q"], [], [[]])

    def test_incomplete_conversation_is_detected_before_selection(self):
        row = {
            "turns": [{"interaction_id": "a"}, {"interaction_id": "b"}],
            "answers": {"interaction_id": ["a", "b"], "ans_full": ["ok", ""]},
        }
        valid, reason = has_complete_answers(row)
        self.assertFalse(valid)
        self.assertEqual(reason, "missing_answer:b")

    def test_requested_five_selects_five_valid_conversations(self):
        valid_row = {
            "session_id": "valid",
            "turns": [{"interaction_id": "a"}],
            "answers": {"interaction_id": ["a"], "ans_full": ["ok"]},
        }
        invalid_row = {
            "session_id": "invalid",
            "turns": [{"interaction_id": "b"}],
            "answers": {"interaction_id": ["b"], "ans_full": [""]},
        }
        indices, excluded = valid_conversation_indices(
            [valid_row, invalid_row, valid_row, valid_row, valid_row, valid_row],
            5,
        )
        self.assertEqual(indices, [0, 2, 3, 4, 5])
        self.assertEqual(len(excluded), 1)

    def test_semantic_shortcut_rejects_opposite_polarity(self):
        self.assertFalse(
            semantic_shortcut(
                "Can it fit?",
                "No, it cannot fit inside the shell.",
                "Yes, it can fit inside the shell.",
            )
        )

    def test_judge_prompt_requires_entity_and_polarity_alignment(self):
        prompt = build_deepseek_judge_prompt("Can it fit?", "No, it cannot.", "Yes, it can.")
        self.assertIn("Yes/No 极性必须一致", prompt)
        self.assertIn('"accuracy": true', prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
