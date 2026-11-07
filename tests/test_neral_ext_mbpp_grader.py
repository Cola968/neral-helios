"""RED/GREEN MBPP isolated Docker scorer contracts. Host never executes code."""
import json, re, subprocess, unittest
from pathlib import Path
from unittest.mock import patch
from scripts.neral_ext_mbpp_grader import prepare_candidate, grade_one, score_saved_mbpp

CASE={"id":"A-mbpp-000","domain":"mbpp","kind":"python","prompt":"Implement add","answer":"def add(x): return x+1"}
META={"id":"A-mbpp-000","stratum":"A","source":"mbpp:test:0","tests":["assert add(1)==2","assert add(10)==11"],"test_setup_code":"","sandbox_required":True}
GOOD={"text":"def add(x):\n return x+1\n","complete":True}


def fake_docker(argv, *, capture_output, timeout, check):
    if argv[:3]==["docker","image","inspect"]:
        return subprocess.CompletedProcess(argv,0,b"[]",b"")
    assert argv[:2]==["docker","run"]
    for flag,val in [("--network","none"),("--pull","never"),("--cap-drop","ALL"),
                     ("--user","65534:65534"),("--security-opt","no-new-privileges")]:
        assert argv[argv.index(flag)+1] == val
    assert "--read-only" in argv
    assert "--pids-limit" in argv
    assert "python:3.11-alpine" in argv
    mount=argv[argv.index("--mount")+1]
    root=Path(mount.split("src=",1)[1].split(",dst=",1)[0])
    marker=re.search(r"MBPP_DONE_[0-9a-f]{32}",(root/"runner.py").read_text()).group()
    return subprocess.CompletedProcess(argv,0,(marker+"\n").encode(),b"")


class MBPPTests(unittest.TestCase):
    def test_valid_markdown_code(self):
        block=chr(96)*3
        self.assertEqual(prepare_candidate("Solution:\n"+block+"python\n"+GOOD["text"]+block),GOOD["text"].strip()+"\n")

    def test_bad_markdown_and_many_blocks(self):
        block=chr(96)*3
        self.assertIsNone(prepare_candidate(block+"python\ndef f(): pass"))
        self.assertIsNone(prepare_candidate(block+"python\nx=1\n"+block+"\n"+block+"python\ny=2\n"+block))

    def test_syntax_error_never_runs_docker(self):
        with patch("subprocess.run",side_effect=AssertionError("host exec")):
            self.assertEqual(grade_one(CASE,META,{"text":"def f(:","complete":True})["status"],"invalid_python")

    def test_incomplete_is_censored_not_wrong(self):
        with patch("subprocess.run",side_effect=AssertionError("host exec")):
            self.assertEqual(grade_one(CASE,META,{"text":"","complete":False})["status"],"censored")

    def test_docker_is_offline_and_exact_tests(self):
        with patch("subprocess.run",side_effect=fake_docker) as spy:
            result=grade_one(CASE,META,GOOD)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(spy.call_count,2)

    def test_docker_timeout_fails_closed(self):
        def expire(argv,*,capture_output,timeout,check):
            if argv[:2]==["docker","image"]:
                return subprocess.CompletedProcess(argv,0,b"[]",b"")
            raise subprocess.TimeoutExpired(argv,timeout)
        with patch("subprocess.run",side_effect=expire):
            self.assertEqual(grade_one(CASE,META,GOOD)["status"],"timeout")

    def test_missing_image_unavailable_not_zero(self):
        with patch("subprocess.run",return_value=subprocess.CompletedProcess([],1,b"",b"")):
            self.assertEqual(grade_one(CASE,META,GOOD)["status"],"unavailable")

    def test_no_completion_marker_no_pass(self):
        def no_marker(argv,*,capture_output,timeout,check):
            return subprocess.CompletedProcess(argv,0,b"[]" if argv[:2]==["docker","image"] else b"done",b"")
        with patch("subprocess.run",side_effect=no_marker):
            self.assertEqual(grade_one(CASE,META,GOOD)["status"],"failed")

    def test_manifest_identity_and_kind_required(self):
        with self.assertRaises(ValueError):
            grade_one(CASE,{**META,"id":"A-mbpp-999"},GOOD)
        with self.assertRaises(ValueError):
            grade_one({**CASE,"kind":"integer"},META,GOOD)

    def test_setup_is_only_ever_executed_in_docker(self):
        meta={**META,"test_setup_code":"import os\nos.system('whoami')"}
        with patch("subprocess.run",side_effect=fake_docker):
            self.assertEqual(grade_one(CASE,meta,GOOD)["status"],"passed")

    def test_report_scalar_counts_preserve_censored(self):
        cases=[CASE]; meta=[META]
        variants={"base":{CASE["id"]:GOOD},
                  "base_fewshot":{CASE["id"]:{"text":"","complete":False}},
                  "neral_v5":{CASE["id"]:{"text":"not a python function","complete":True}}}
        with patch("subprocess.run",side_effect=fake_docker):
            report=score_saved_mbpp(cases,meta,variants)
        self.assertEqual(report["by_variant"]["base"]["passed"],1)
        self.assertEqual(report["by_variant"]["base_fewshot"]["censored"],1)
        self.assertEqual(report["by_variant"]["neral_v5"]["no_code"],1)
        self.assertFalse(report["release_validated"])
        dump=json.dumps(report)
        for word in ("def add","Implement add","assert add","test_setup_code"):
            self.assertNotIn(word,dump)

    def test_untrusted_source_not_passed_to_docker_argv(self):
        with patch("subprocess.run",side_effect=fake_docker) as spy:
            grade_one(CASE,META,{"text":"def f():\n __import__('os').system('id')","complete":True})
        command=spy.call_args_list[-1].args[0]
        self.assertNotIn("os.system",str(command))


if __name__=="__main__":
    unittest.main()
