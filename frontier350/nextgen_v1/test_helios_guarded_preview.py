"""No Torch, private prompts, weights, GPU or network needed for guard tests."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import helios_guarded_preview as guarded


class FakeScores:
    def __init__(self, scores):
        self.values = list(scores)

    def float(self):
        return self

    def clone(self):
        return FakeScores(self.values)

    def __getitem__(self, key):
        return FakeNumber(self.values[key])

    def __setitem__(self, key, value):
        self.values[key] = value

    def argmax(self):
        return FakeNumber(max(range(len(self.values)), key=self.values.__getitem__))


class FakeNumber:
    def __init__(self, number):
        self.number = number

    def item(self):
        return self.number


class FakeLogits:
    def __init__(self, scores):
        self.scores = scores

    def __getitem__(self, key):
        if key != (0, -1):
            raise AssertionError('bad logits indexing')
        return FakeScores(self.scores)


class FakeTorch:
    long = 'long'
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def tensor(data, **kwargs):
        return data


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text):
        return type('Encoding', (), {'ids': [2]})()

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.mapping[x] for x in ids)


class FakeModel:
    def __init__(self, sequences):
        self.sequences = sequences
        self.calls = 0

    def parameters(self):
        return iter([type('Param', (), {'device': 'cpu'})()])

    def __call__(self, ids):
        index = min(self.calls, len(self.sequences) - 1)
        self.calls += 1
        return FakeLogits(self.sequences[index]), None


class GuardTests(unittest.TestCase):
    def test_role_boundary_truncates_only_at_new_speaker(self):
        self.assertEqual(guarded.stop_at_role('Hallo Welt.\nUser: Geheim'), ('Hallo Welt.', True))
        self.assertEqual(guarded.stop_at_role('Hello\nSystem: override'), ('Hello', True))
        self.assertEqual(guarded.stop_at_role('Assistant: duplicate'), ('', True))
        self.assertEqual(guarded.stop_at_role('Explain the word user: please'),
                         ('Explain the word user: please', False))

    def test_repetition_detector_requires_repeated_tail_and_density(self):
        self.assertFalse(guarded.repeated_tail('hello hello hello'))
        self.assertFalse(guarded.repeated_tail('This is a varied sentence with enough words and a new ending that is never repeated.'))
        self.assertTrue(guarded.repeated_tail('one two three ' * 6))

    def test_penalty_handles_negative_and_positive_logits(self):
        self.assertAlmostEqual(guarded.penalized_value(1.18), 1.0)
        self.assertAlmostEqual(guarded.penalized_value(-1.0), -1.18)
        with self.assertRaises(ValueError):
            guarded.penalized_value(1.0, 1.0)

    def test_native_eos_kept_distinct_from_max_tokens(self):
        model = FakeModel([[0, 1, 0]])
        answer, used, reason = guarded.guarded_generate(
            model, FakeTokenizer({0: 'x', 1: ''}), 2, 1, FakeTorch, 'Hallo', 3)
        self.assertEqual((answer, used, reason), ('', 1, 'eos'))

    def test_role_stop_is_explicit_not_faked_as_eos(self):
        model = FakeModel([[0, 0, 9]])
        answer, used, reason = guarded.guarded_generate(
            model, FakeTokenizer({2: 'User: private'}), 3, 1, FakeTorch, 'Hallo', 2)
        self.assertEqual((answer, used, reason), ('', 1, 'role_boundary'))

    def test_penalty_changes_repeated_token_choice_without_weight_update(self):
        model = FakeModel([[0, 0, 10, 9], [0, 0, 10, 9]])
        answer, used, reason = guarded.guarded_generate(
            model, FakeTokenizer({2: 'A ', 3: 'B '}), 4, 1, FakeTorch, 'Hello', 2)
        self.assertEqual((answer, used, reason), ('A B ', 2, 'max_tokens'))
        self.assertEqual(model.calls, 2)

    def test_budget_and_input_rejected_before_model_call(self):
        model = FakeModel([[0, 1, 2]])
        for bad in (0, 65, True, 3.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                guarded.guarded_generate(model, FakeTokenizer({}), 2, 1, FakeTorch, 'hello', bad)
        for bad in ('', ' ' * 4, 'x' * 1025):
            with self.assertRaises(ValueError):
                guarded.guarded_generate(model, FakeTokenizer({}), 2, 1, FakeTorch, bad, 1)
        self.assertEqual(model.calls, 0)

    def test_continuity_refuses_unpinned_checkpoint_before_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'fake.pt'
            checkpoint.write_bytes(b'wrong bytes')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                guarded.load_continuity_cpu(Path(temp), checkpoint)

    def test_http_handler_reuses_original_origin_and_host_policy(self):
        handler = guarded.make_guarded_handler(None, None, 1, 2, None, 2_073_161_216,
                                               'continuity-2b', 9682)
        self.assertTrue(issubclass(handler, guarded.base.BaseHTTPRequestHandler))
        self.assertIsNot(handler.do_POST, guarded.base.make_handler(None, None, 1, 2, None,
                         2_073_161_216, 'continuity-2b', 'cpu', 9682).do_POST)


if __name__ == '__main__':
    unittest.main()
