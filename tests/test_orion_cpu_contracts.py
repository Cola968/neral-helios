import json
from pathlib import Path

import pytest

from checkpoint_io_v3 import (
    FORMAT,
    load_verified_checkpoint,
    make_resume_payload,
    resolve_last_good,
    stable_hash,
    validate_resume_payload,
    verify_checkpoint_file,
    save_checkpoint_atomic,
)
from orion_static_metrics import MIB, SPECS, build_report, kv_cache_bytes, linear_gemm_calls, parameter_partition


def test_all_static_parameter_contracts_are_exact():
    expected = {
        "50m_legacy": 50_012_976,
        "50m_fused": 50_012_976,
        "helix_a1": 99_783_168,
        "helix_a2": 100_470_912,
        "orion_a3": 100_002_304,
        "orion_a4": 100_103_936,
    }
    for name, value in expected.items():
        assert parameter_partition(SPECS[name])["total"] == value


def test_structural_gemm_contracts():
    assert linear_gemm_calls(SPECS["50m_legacy"])["per_forward"] == 40
    assert linear_gemm_calls(SPECS["50m_fused"])["per_forward"] == 32
    assert linear_gemm_calls(SPECS["helix_a1"])["per_forward"] == 224
    assert linear_gemm_calls(SPECS["helix_a2"])["per_forward"] == 168
    assert linear_gemm_calls(SPECS["orion_a3"])["per_forward"] == 56
    assert linear_gemm_calls(SPECS["orion_a4"])["per_forward"] == 64


def test_compact_kv_contracts_at_4k():
    assert kv_cache_bytes(SPECS["50m_legacy"], 4096) == 78 * MIB
    assert kv_cache_bytes(SPECS["helix_a1"], 4096) == 64 * MIB
    assert kv_cache_bytes(SPECS["helix_a2"], 4096) == 48 * MIB
    assert kv_cache_bytes(SPECS["orion_a3"], 4096) == 28 * MIB
    assert kv_cache_bytes(SPECS["orion_a4"], 4096) == 32 * MIB


def test_static_report_never_claims_gpu_or_quality_measurement():
    r = build_report()
    assert r["gpu_measurements_present"] is False
    assert r["quality_scores_for_untrained_orion_present"] is False
    assert r["models"]["orion_a3"]["evidence_class"]["training_vram_range"] == "planning-heuristic-not-measured"


def _payload(torch):
    return make_resume_payload(
        model_state={"w": torch.arange(4, dtype=torch.float32)},
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state={"last_epoch": 7},
        step=8,
        token_count=65536,
        curriculum_phase="orion-pretrain-s1",
        dataset_cursor={"shard": 3, "document": 19, "token_offset": 0},
        config={"variant": "orion_a3", "seq": 512},
        architecture_contract={"parameters": 100_002_304, "name": "ORION A3"},
        source_commit="abcdef1234567890",
        dataset_manifest={"revision": "1234567890abcdef", "shards": ["x"]},
        rng_state={"python_random": __import__("random").getstate(), "numpy_random": None, "torch_cpu": torch.get_rng_state(), "torch_cuda": []},
    )


def test_resume_payload_hashes_detect_drift():
    torch = pytest.importorskip("torch")
    payload = _payload(torch)
    assert payload["format"] == FORMAT
    validate_resume_payload(payload)
    payload["config"]["seq"] = 1024
    with pytest.raises(ValueError, match="config hash mismatch"):
        validate_resume_payload(payload)


def test_atomic_checkpoint_roundtrip_and_corruption_detection(tmp_path: Path):
    torch = pytest.importorskip("torch")
    target = tmp_path / "checkpoint.pt"
    result = save_checkpoint_atomic(_payload(torch), target)
    assert result["sha256"] == verify_checkpoint_file(target)["sha256"]
    assert resolve_last_good(tmp_path) == target.resolve()
    loaded = load_verified_checkpoint(target, trusted_local=True)
    assert loaded["metadata"]["step"] == 8

    raw = target.read_bytes()
    target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint_file(target)


def test_checkpoint_deserialization_is_fail_closed(tmp_path: Path):
    torch = pytest.importorskip("torch")
    target = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic(_payload(torch), target)
    with pytest.raises(ValueError, match="trusted_local"):
        load_verified_checkpoint(target)


def test_last_good_pointer_cannot_escape_directory(tmp_path: Path):
    pointer = tmp_path / "last_good_checkpoint.json"
    pointer.write_text(json.dumps({"path": "../escape.pt", "sha256": stable_hash("x")}), encoding="utf8")
    with pytest.raises(ValueError, match="escapes"):
        resolve_last_good(tmp_path)
