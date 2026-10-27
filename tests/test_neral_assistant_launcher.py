"""Regression tests: fail closed when Ollama is absent or an old NERAL app owns the port."""
import unittest

from neral_assistant_launcher import (
    LauncherError, assistant_environment, installed_ollama_model, assistant_ready,
)


class LauncherTests(unittest.TestCase):
    def test_only_explicit_local_model_is_ready(self):
        def local_json(url):
            self.assertEqual(url, "http://127.0.0.1:11434/api/tags")
            return {"models":[{"name":"qwen3.5:9b"},{"name":"qwen3.5:4b"}]}
        self.assertTrue(installed_ollama_model(local_json))
        self.assertFalse(installed_ollama_model(lambda _:{"models":[]}))
        self.assertFalse(installed_ollama_model(lambda _:{"models":[{"name":"qwen3.5:4b"}]}))

    def test_ollama_bad_response_fails_closed(self):
        for data in ({}, [], {"models":"fake"}, {"models":[{"name":5}]}, {"models":None}):
            with self.subTest(data=data),self.assertRaises(LauncherError):
                installed_ollama_model(lambda _:data)

    def test_old_neral_port_does_not_count_as_assistant(self):
        fetch=lambda _:{"name":"Neral / Helios","assistant":{"enabled":False,"base_model":"qwen3.5:9b"}}
        self.assertFalse(assistant_ready(fetch))
        fetch=lambda _:{"name":"Neral / Helios","assistant":{"enabled":True,"base_model":"qwen3.5:9b"}}
        self.assertTrue(assistant_ready(fetch))
        self.assertFalse(assistant_ready(lambda _:{"name":"some app","assistant":{"enabled":True,"base_model":"qwen3.5:9b"}}))

    def test_assistant_env_is_explicit_and_keeps_own_model_optional(self):
        env=assistant_environment({"HELIOS_DEVICE":"cuda","NERAL_ASSISTANT_ENABLED":"0","HELIOS_PREPARATION_MODE":"1"})
        self.assertEqual(env["NERAL_ASSISTANT_ENABLED"],"1")
        self.assertEqual(env["NERAL_ASSISTANT_MODEL"],"qwen3.5:9b")
        self.assertEqual(env["HELIOS_PREPARATION_MODE"],"0")
        self.assertEqual(env["HELIOS_DEVICE"],"cpu")
        self.assertEqual(env["HELIOS_PORT"],"9680")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"")

if __name__=="__main__":
    unittest.main()
