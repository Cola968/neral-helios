"""CPU RED/GREEN: sealed SHA-checked Qwen/v5/v7 logic evaluation contracts."""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_v7_logic_pilot import build_logic_experiment,evaluation_cases
from scripts import neral_v7_logic_compare as compare
from scripts import neral_v6_fresh_dev_eval as old


def _candidate(directory:Path):
    directory.mkdir()
    weights=directory/"adapter_model.safetensors"
    weights.write_bytes(b"test-only-not-real-weights")
    (directory/"adapter_config.json").write_text("{}",encoding="utf8")
    _,_,meta=build_logic_experiment()
    receipt={
        "status":compare.CANDIDATE_STATUS,
        "steps_completed":120,
        "train_rows":800,"validation_rows":128,
        "checkpoint_mode":"best_validation_loss",
        "checkpoint_interval":20,
        "checkpoint_selected_step":80,
        "source_counts":{"original_replay":560,"new_logic_verified":240},
        "train_sha256":meta["train_sha256"],
        "validation_sha256":meta["validation_sha256"],
        "candidate_adapter_sha256":sha256(weights.read_bytes()).hexdigest(),
        "predecessor_adapter_sha256":old.V5_SHA,
        "release_validated":False,"adapter_promoted":False,
        "independent_transfer_proven":False,
    }
    (directory.parent/"receipt.json").write_text(json.dumps(receipt),encoding="utf8")
    return directory.parent/"receipt.json"


def _answers(cases,wrong_indices=()):
    return [dict(id=c["id"],text=("FINAL_ANSWER: 999999" if i in wrong_indices
                                else "FINAL_ANSWER: "+c["answer"]),
                 complete=True,new_tokens=6)
            for i,c in enumerate(cases)]


def _write_jsonl(path,rows):
    with path.open("x",encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")


class LogicWeightEvalContracts(unittest.TestCase):
    def test_identical_no_tool_messages_for_all_model_arms(self):
        case=evaluation_cases()[0]
        msgs=[compare.shared_messages(case,compare.SOURCE_VARIANTS[arm])
              for arm in compare.ARMS]
        self.assertEqual(msgs[0],msgs[1])
        self.assertEqual(msgs[1],msgs[2])
        self.assertNotIn("TOOL",str(msgs))
        with self.assertRaises(ValueError):
            compare.shared_messages(case,"user_owned_new_arm")

    def test_prepare_sealed_private_72_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"new"
            info=compare.prepare_run(path)
            self.assertEqual(info["retention_count"],24)
            self.assertEqual(info["transfer_count"],48)
            self.assertEqual(len(compare._cases_folder(path)),72)
            with self.assertRaises(ValueError):
                compare.prepare_run(path)
            cases=(path/"cases.jsonl").read_text("utf8").splitlines()
            cases.reverse()
            (path/"cases.jsonl").write_text("\n".join(cases)+"\n",encoding="utf8")
            with self.assertRaises(ValueError):
                compare._cases_folder(path)

    def test_verified_weight_receipt_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            adapter=root/"adapter"
            receipt=_candidate(adapter)
            result=compare.candidate_receipt(receipt,adapter)
            self.assertEqual(len(result["candidate_sha256"]),64)
            proof=json.loads(receipt.read_text("utf8"))
            proof["validation_rows"]=127
            receipt.write_text(json.dumps(proof),encoding="utf8")
            with self.assertRaises(ValueError):
                compare.candidate_receipt(receipt,adapter)

    def test_score_true_paired_changes_and_private_scalar_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/"pilot"
            compare.prepare_run(folder)
            cases=compare._cases_folder(folder)
            receipt=_candidate(root/"adapter")
            _write_jsonl(folder/"qwen.jsonl",_answers(cases,wrong_indices=(0,1,24,25,26)))
            _write_jsonl(folder/"v5.jsonl",_answers(cases,wrong_indices=(0,1,24,25)))
            _write_jsonl(folder/"candidate.jsonl",_answers(cases,wrong_indices=(0,24)))
            report=compare.combine(folder,receipt,root/"adapter")
            self.assertEqual(report["status"],compare.REPORT_STATUS)
            self.assertEqual(report["slices"]["retention"]["paired"]["candidate_minus_v5"]["improvements"],1)
            self.assertEqual(report["slices"]["transfer"]["paired"]["candidate_minus_v5"]["improvements"],1)
            self.assertEqual(report["slices"]["transfer"]["by_arm"]["candidate"]["correct"],47)
            self.assertFalse(report["release_validated"])
            self.assertFalse(report["quality_improvement_proven"])
            self.assertFalse(report["adapter_promoted"])
            self.assertTrue(report["model_only_no_tools"])
            serialized=json.dumps(report,ensure_ascii=False)
            self.assertNotIn(cases[0]["prompt"],serialized)
            self.assertNotIn(cases[0]["id"],serialized)
            with self.assertRaises(ValueError):
                compare.combine(folder,receipt,root/"adapter")

    def test_reordered_response_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"new"
            compare.prepare_run(path)
            cases=compare._cases_folder(path)
            responses=_answers(cases)
            responses[0],responses[1]=responses[1],responses[0]
            with self.assertRaises(ValueError):
                compare._grades(cases,responses)

    def test_limited_sample_not_a_release_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/"private"
            compare.prepare_run(folder)
            cases=compare._cases_folder(folder)
            receipt=_candidate(root/"adapter")
            for arm in compare.ARMS:
                _write_jsonl(folder/(arm+".jsonl"),_answers(cases))
            report=compare.combine(folder,receipt,root/"adapter")
            for role in ("retention","transfer"):
                for value in report["slices"][role]["paired"].values():
                    self.assertEqual(value["improvements"],0)
                    self.assertEqual(value["regressions"],0)
                    self.assertEqual(value["exact_mcnemar_p_exploratory"],1.)
            self.assertFalse(report["quality_improvement_proven"])


if __name__=="__main__":
    unittest.main()
