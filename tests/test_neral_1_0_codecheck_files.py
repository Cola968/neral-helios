"""TDD: NERAL 1.0 Python syntax gate must never execute user code."""
from unittest.mock import patch
import unittest
from neral_preview_server import create_preview_app

HOST="http://127.0.0.1:9690"


class Stub:
    model="qwen3:8b"
    def respond_chat(self,messages):
        return "OK"
    def respond_think(self,messages):
        return "OK"


class CodeCheckContracts(unittest.TestCase):
    def setUp(self):
        self.client=create_preview_app(Stub(),secret_key=b"0"*32).test_client()

    def token(self):
        self.client.get("/",base_url=HOST)
        reply=self.client.get("/api/neral/preview/session",base_url=HOST)
        return {"X-Neral-CSRF":reply.json["csrf"]}

    def test_python_syntax_checker_only_parses_and_never_executes(self):
        hdr=self.token()
        with patch("os.system",side_effect=AssertionError("Code executed")):
            out=self.client.post("/api/neral/preview/codecheck",base_url=HOST,
                 headers=hdr,json={"source":"import os\nos.system('should-never-run')"})
        self.assertEqual(out.status_code,200)
        self.assertTrue(out.json["syntax_ok"])
        self.assertIs(out.json["code_executed"],False)
        self.assertEqual(out.json["scope"],"python_ast_only")

    def test_syntax_error_reports_a_bounded_location_no_code(self):
        out=self.client.post("/api/neral/preview/codecheck",base_url=HOST,
            headers=self.token(),json={"source":"def func(:\n    pass"})
        self.assertEqual(out.status_code,200)
        self.assertFalse(out.json["syntax_ok"])
        self.assertEqual(out.json["line"],1)
        self.assertFalse(out.json["code_executed"])
        self.assertNotIn("traceback",str(out.json).lower())

    def test_ast_valid_but_python_compiler_invalid_must_fail(self):
        # ast.parse() alone wrongly accepts these unusable Python programs.
        headers = self.token()
        invalid = (
            "return 42",
            "break",
            "continue",
            "nonlocal x",
            "await sleep(1)",
        )
        for source in invalid:
            with self.subTest(source=source):
                out = self.client.post("/api/neral/preview/codecheck",
                    base_url=HOST, headers=headers, json={"source": source})
                self.assertEqual(out.status_code, 200)
                self.assertFalse(out.json["syntax_ok"], source)
                self.assertIs(out.json["code_executed"], False)
                self.assertGreaterEqual(out.json["line"], 1)

    def test_python_compiler_check_does_not_run_source(self):
        headers = self.token()
        dangerous = "print('must-not-execute')"
        with patch("builtins.print", side_effect=AssertionError("Python source ran")):
            out = self.client.post("/api/neral/preview/codecheck",
                base_url=HOST, headers=headers, json={"source": dangerous})
        self.assertEqual(out.status_code, 200)
        self.assertTrue(out.json["syntax_ok"])
        self.assertIs(out.json["code_executed"], False)

    def test_reject_malformed_large_and_cross_origin(self):
        hdr=self.token()
        for payload in ({"source":""},{"source":"x"*5000},
                        {"source":"print(1)","command":"run"},
                        {"filename":"abc.py"},[],{"source":4}):
            out=self.client.post("/api/neral/preview/codecheck",base_url=HOST,
                 headers=hdr,json=payload)
            self.assertEqual(out.status_code,400)
        out=self.client.post("/api/neral/preview/codecheck",base_url=HOST,
           headers={**hdr,"Origin":"https://evil.invalid"},json={"source":"x=1"})
        self.assertEqual(out.status_code,403)

    def test_capabilities_distinguish_syntax_check_from_code_execution(self):
        self.token()
        out=self.client.get("/api/neral/preview/plugins",base_url=HOST)
        self.assertTrue(out.json["plugins"]["python_syntax_check"]["available"])
        self.assertFalse(out.json["plugins"]["local_code_execution"]["available"])
        self.assertTrue(out.json["plugins"]["selected_file_context"]["available"])


if __name__=="__main__":
    unittest.main()
