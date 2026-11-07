"""Real isolated Docker smoke test with synthetic code only (no NERAL data)."""
from __future__ import annotations

import os
import unittest
from scripts.neral_ext_mbpp_grader import grade_one

CASE={"id":"A-mbpp-000","domain":"mbpp","kind":"python",
      "prompt":"Write add(x)","answer":"def add(x): return x+1"}
META={"id":"A-mbpp-000","stratum":"A","source":"mbpp:test:0",
      "tests":["assert add(1) == 2", "assert add(17) == 18"],
      "test_setup_code":"","sandbox_required":True}


@unittest.skipUnless(os.environ.get("NERAL_MBPP_LIVE_DOCKER")=="1",
                     "Live Docker smoke is opt-in CI only")
class LiveDockerTests(unittest.TestCase):
    def check_code(self, code):
        return grade_one(CASE,META,{"text":code,"complete":True})["status"]

    def test_good_function_passes_actual_docker(self):
        self.assertEqual(self.check_code("def add(x):\n    return x+1"),"passed")

    def test_wrong_function_fails_actual_docker(self):
        self.assertEqual(self.check_code("def add(x):\n    return x+2"),"failed")

    def test_early_exit_zero_does_not_look_like_success(self):
        self.assertEqual(self.check_code(
            "def add(x):\n    import os\n    os._exit(0)"),"failed")

    def test_network_is_blocked_inside_docker(self):
        result=self.check_code(
            "def add(x):\n    import socket\n"
            "    socket.create_connection(('1.1.1.1',80),timeout=0.4)\n"
            "    return x+1")
        self.assertIn(result,("failed","timeout"))


if __name__=="__main__":
    unittest.main()
