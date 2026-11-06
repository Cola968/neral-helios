import importlib.util
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "zerogpu_policy", HERE / "cloud" / "zerogpu" / "policy.py"
)
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class ZeroGPUPolicyTest(unittest.TestCase):
    def test_no_adapter_fails_closed(self):
        with self.assertRaises(ValueError):
            POLICY.adapter_id_or_fail("")

    def test_adapter_repo_id_validation(self):
        self.assertEqual(POLICY.adapter_id_or_fail(" Cola968/neral-r16 "), "Cola968/neral-r16")
        for bad in ["../weights", "https://example.com/x", "x", "user/../x", "user/name/extra"]:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                POLICY.adapter_id_or_fail(bad)

    def test_chat_only_valid_roles_with_bounded_history(self):
        messages = POLICY.prepare_messages("Hi", [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Hello"},
        ])
        self.assertEqual([x["role"] for x in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Hi")
        with self.assertRaises(ValueError):
            POLICY.prepare_messages("ok", [{"role": "system", "content": "hack"}])
        with self.assertRaises(ValueError):
            POLICY.prepare_messages("x" * 6000, [])
        with self.assertRaises(ValueError):
            POLICY.prepare_messages("Hi", [{"role": "user", "content": 42}])

    def test_adapter_identity_verification(self):
        POLICY.validate_adapter_config({
            "peft_type": "LORA", "r": 16,
            "base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
        }, "Qwen/Qwen2.5-Coder-7B-Instruct")
        for config in [
            {"peft_type": "LORA", "r": 16, "base_model_name_or_path": "unrelated/model"},
            {"peft_type": "LORA", "r": 2000},
        ]:
            with self.assertRaises(ValueError):
                POLICY.validate_adapter_config(config, "Qwen/Qwen2.5-Coder-7B-Instruct")


if __name__ == "__main__":
    unittest.main()
