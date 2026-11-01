"""Portable NERAL Arctus350 checkpoint-selection self-test.

Runs only synthetic 400-case reports against the real selector. Pure Python
stdlib; never loads weights, performs HTTP requests, or trains on a GPU.
This is a focused diagnostic, NOT a replacement for the complete test suite.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

try:
    from frontier350 import select_bilingual_recovery_v6 as selector
except ModuleNotFoundError:
    import select_bilingual_recovery_v6 as selector


def _write(folder: Path, name: str, data: dict) -> Path:
    path=folder/(name+".json")
    path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf8")
    return path


def _fixture(folder: Path):
    sha="a"*64
    ck="b"*64
    baseline={
        "suite":selector.SUITE,"cases_sha256":sha,"cases":400,
        "overall_accuracy_pct":45.0,
        "directions":{
            "de_to_en":{"correct":80,"total":200,"accuracy_pct":40.0},
            "en_to_de":{"correct":100,"total":200,"accuracy_pct":50.0},
        },
    }
    holdout={
        "suite":selector.SUITE,"cases_sha256":sha,
        "checkpoint_sha256":ck,"cases":400,
        "overall_accuracy_pct":47.0,
        "directions":{
            "de_to_en":{"correct":88,"total":200,"accuracy_pct":44.0},
            "en_to_de":{"correct":100,"total":200,"accuracy_pct":50.0},
        },
    }
    gate={
        "selection_allowed":True,"selected":"candidate-A",
        "candidates":[{
            "name":"candidate-A","pass":True,
            "checkpoint_sha256":ck,"validation_loss":1.2,
            "overall_accuracy_pct":65.0,
            "categories":{"bilingual":55.0},
        }],
    }
    return ( _write(folder,"baseline",baseline),
             _write(folder,"gate",gate),
             _write(folder,"holdout",holdout) )


def _select(baseline: Path, candidates: list[tuple[str,Path,Path]], out: Path):
    args=["select","--baseline-holdout",str(baseline)]
    for name,gate,holdout in candidates:
        args+=["--candidate",name,str(gate),str(holdout),str(out.parent/("ck-"+name))]
    args+=["--out",str(out)]
    with patch.object(sys,"argv",args), contextlib.redirect_stdout(io.StringIO()):
        selector.main()
    return json.loads(out.read_text(encoding="utf8"))


def main() -> int:
    passed=0
    with tempfile.TemporaryDirectory(prefix="arctus-stdlib-check-") as directory:
        root=Path(directory)
        baseline,gate,holdout=_fixture(root)
        out=root/"decision.json"
        report=_select(baseline,[("candidate-A",gate,holdout)],out)
        assert report["selected"]=="candidate-A" and report["selection_allowed"]
        passed+=1

        wrong_gate=json.loads(gate.read_text())
        wrong_gate["selected"]="candidate-B"
        gate_bad=_write(root,"bad-gate",wrong_gate)
        report=_select(baseline,[("candidate-A",gate_bad,holdout)],out)
        assert not report["selection_allowed"]
        assert "gate_candidate_identity_mismatch" in report["candidates"][0]["failures"]
        passed+=1

        wrong_ck=json.loads(holdout.read_text())
        wrong_ck["checkpoint_sha256"]="c"*64
        ck_bad=_write(root,"bad-holdout-checkpoint",wrong_ck)
        report=_select(baseline,[("candidate-A",gate,ck_bad)],out)
        assert not report["selection_allowed"]
        assert "checkpoint_identity_mismatch" in report["candidates"][0]["failures"]
        passed+=1

        invalid=json.loads(holdout.read_text())
        invalid["overall_accuracy_pct"]=float("nan")
        nan_holdout=_write(root,"nan-holdout",invalid)
        report=_select(baseline,[("candidate-A",gate,nan_holdout)],out)
        assert not report["selection_allowed"]
        assert "invalid_holdout_metric" in report["candidates"][0]["failures"]
        passed+=1

        inflated=json.loads(holdout.read_text())
        inflated["overall_accuracy_pct"]=99.0
        inflated_holdout=_write(root,"inflated-holdout",inflated)
        report=_select(baseline,[("candidate-A",gate,inflated_holdout)],out)
        assert not report["selection_allowed"]
        assert "invalid_holdout_metric" in report["candidates"][0]["failures"]
        passed+=1

        competing=json.loads(gate.read_text())
        competing["selected"]="candidate-B"
        competing["candidates"][0].update(
            name="candidate-B",checkpoint_sha256="c"*64,
            overall_accuracy_pct=55.0)
        second_gate=_write(root,"candidate-B-gate",competing)
        better_holdout=json.loads(holdout.read_text())
        better_holdout["checkpoint_sha256"]="c"*64
        better_holdout["overall_accuracy_pct"]=52.0
        for entry in better_holdout["directions"].values():
            entry.update(correct=104,accuracy_pct=52.0)
        second_holdout=_write(root,"candidate-B-holdout",better_holdout)
        report=_select(baseline,[
            ("candidate-A",gate,holdout),
            ("candidate-B",second_gate,second_holdout)],out)
        assert report["selected"]=="candidate-A"
        assert all(row["pass"] for row in report["candidates"])
        passed+=1

        corrupt=json.loads(baseline.read_text())
        corrupt["directions"]["de_to_en"]["correct"]=199
        corrupt_baseline=_write(root,"corrupt-baseline",corrupt)
        try:
            _select(corrupt_baseline,[("candidate-A",gate,holdout)],out)
        except RuntimeError as exc:
            assert "baseline" in str(exc)
        else:
            raise AssertionError("Invalid baseline was accepted")
        passed+=1

    print(json.dumps({
        "state":"PASS_OFFLINE","tests_passed":passed,
        "model_loaded":False,"training_started":False,
        "network_requested":False,"paid_gpu_used":False,
        "limitation":"Only bounded selector logic, not complete model/runtime CI.",
    },ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
