"""CPU-only guardrail for recovering the single interrupted raw50 leg."""
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from scripts.neral_v43_reference50 import generate_cases, seal_cases, FROZEN
from scripts.neral_v43_reference50_live import verify_recovery_preflight


class Reference50RawRecoveryCPU(unittest.TestCase):
    def test_preserved_complete_neral50_can_resume_raw_only_once(self):
        cases=generate_cases("b9"*32)
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/"v43-reference50"/"45264946953"
            folder.mkdir(parents=True)
            digest=seal_cases(cases,folder/"sealed50.jsonl",folder/"manifest.json")
            with (folder/"neral10-50.jsonl").open("x",encoding="utf-8") as stream:
                for c in cases:
                    row={"id":c["id"],"category":c["category"],
                         "engine":"neral10_obsidian","source_commit":FROZEN,
                         "prompt_sha256":sha256(c["prompt"].encode()).hexdigest(),
                         "grade":"pass","http_status":200,"benchmark_isolated":True,
                         "model_generated":True,"latency_ms":100,
                         "model_text":"OWNER-PRIVATE"}
                    stream.write(json.dumps(row)+"\n")
            self.assertTrue(verify_recovery_preflight(folder,expected_sha=digest))
            # A begun raw run can never be run a second time.
            (folder/"raw9b-50.jsonl").write_text("",encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_recovery_preflight(folder,expected_sha=digest)

    def test_original_http_errors_are_recorded_failures_not_missing_cases(self):
        # Original neral() emits non-200 as grade=fail, reason=http_failure.
        # Recovery must not discard those immutable scored observations.
        cases=generate_cases("ba"*32)
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/"v43-reference50"/"45264946953"
            folder.mkdir(parents=True)
            digest=seal_cases(cases,folder/"sealed50.jsonl",folder/"manifest.json")
            rows=[]
            for i,c in enumerate(cases):
                is_transport_failure=i in (1,17)
                rows.append({"id":c["id"],"category":c["category"],
                            "engine":"neral10_obsidian","source_commit":FROZEN,
                            "prompt_sha256":sha256(c["prompt"].encode()).hexdigest(),
                            "grade":"fail" if is_transport_failure else "pass",
                            "reason":"http_failure" if is_transport_failure else "independent_exact",
                            "http_status":0 if i==1 else (503 if i==17 else 200),
                            "benchmark_isolated":True,"model_generated":False,
                            "latency_ms":100,"model_text":"" if is_transport_failure else "OWNER-PRIVATE"})
            stage=folder/"neral10-50.jsonl"
            with stage.open("x",encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row)+"\n")
            self.assertTrue(verify_recovery_preflight(folder,expected_sha=digest))
            # Any unscored/contradictory error stays fail-closed.
            rows[1]["grade"]="pass"
            stage.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_recovery_preflight(folder,expected_sha=digest)
            rows[1]["grade"]="fail"
            rows[1]["reason"]="independent_exact"
            stage.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_recovery_preflight(folder,expected_sha=digest)

    def test_refuses_unrecognized_source_directory_or_seal(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/"another-run"
            folder.mkdir()
            with self.assertRaises(ValueError):
                verify_recovery_preflight(folder,expected_sha="0"*64)


if __name__=="__main__":
    unittest.main()
