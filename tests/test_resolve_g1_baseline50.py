import json
from pathlib import Path

import pytest

from resolve_g1_baseline50 import REQUIRED, resolve


def make_nested_handoff(tmp_path: Path, sha: str, *, complete: bool = True):
    assets=tmp_path/"assets"
    root=assets/"reports"/"gpu_control"/"34151630663"/"b2-hardened-benchmark-20260907"/"34151630663"/"benchmark"
    b2=root/"b2";b2.mkdir(parents=True)
    for name in REQUIRED:(b2/name).write_text("{}",encoding="utf8")
    handoff={
        "b2_complete":complete,
        "cumulative_tokens":100_000_000,
        "checkpoints":{"b2":{"sha256":sha}},
        "generation_health_gate":{"proceed":False},
        "long_context_gate":{"proceed":False},
    }
    (root/"B2_HANDOFF.json").write_text(json.dumps(handoff),encoding="utf8")
    return assets,root,b2


def test_resolve_finds_recursively_nested_exact_checkpoint(tmp_path):
    sha="a"*64
    assets,root,b2=make_nested_handoff(tmp_path,sha)
    result=resolve(assets,sha)
    assert result["resolved"] is True
    assert Path(result["benchmark_root"])==root.resolve()
    assert Path(result["b2_report_dir"])==b2.resolve()
    assert result["checkpoint_sha256"]==sha
    assert result["weights_modified"] is False


def test_resolve_rejects_mismatched_checkpoint_sha(tmp_path):
    assets,_,_=make_nested_handoff(tmp_path,"a"*64)
    with pytest.raises(FileNotFoundError,match="No hardened B2 benchmark matches"):
        resolve(assets,"b"*64)


def test_resolve_rejects_incomplete_report_set(tmp_path):
    sha="c"*64
    assets,_,b2=make_nested_handoff(tmp_path,sha)
    (b2/"v5.json").unlink()
    with pytest.raises(FileNotFoundError,match="No hardened B2 benchmark matches"):
        resolve(assets,sha)


def test_resolve_rejects_incomplete_b2_handoff(tmp_path):
    sha="d"*64
    assets,_,_=make_nested_handoff(tmp_path,sha,complete=False)
    with pytest.raises(FileNotFoundError,match="No hardened B2 benchmark matches"):
        resolve(assets,sha)
