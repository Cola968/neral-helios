"""RED tests for Vast.ai Arctus350 one-GPU safety preparation. CPU only."""
import hashlib
import json
from pathlib import Path
import pytest

from frontier350.vast4090_guard import (
    make_plan, validate_host, validate_data_manifest, prepare_run, sha256_file
)
from frontier350.chat350_contract import EXPECTED_CONFIG, EXPECTED_TOKENIZER_SHA256 as TOK


def checkpoint_meta(processed=2_073_161_216,target=10_000_000_000):
    return {"config":EXPECTED_CONFIG, "tokenizer_sha256":TOK,
            "checkpoint_schema":2, "processed_tokens":processed,
            "target_tokens":target, "step":126534,
            "model":{"stub":True},"optimizer":{"stub":True},
            "torch_rng":True,"python_rng":True,"numpy_rng":True}


def small_data(tmp_path):
    root=tmp_path/"data"
    for partition in ("train","valid"):
        folder=root/partition
        folder.mkdir(parents=True)
        sample=folder/f"{partition}_00000.bin"
        sample.write_bytes(bytes([1,0])*2048)
    manifest={"schema":2,"architecture":"frontier350-v1",
             "tokenizer":{"tokenizer_sha256":TOK},
             "benchmark_content_included":False}
    for part,field in (("train","train_files"),("valid","valid_files")):
        path=root/part/f"{part}_00000.bin"
        manifest[field]=[{"path":path.name,"sha256":sha256_file(path),"tokens":2048}]
    (root/"manifest.json").write_text(json.dumps(manifest))
    return root


def test_plan_honors_original_target_and_never_resets_training():
    plan=make_plan(checkpoint_meta(),extra_tokens=131072,price_usd=0.366,cap_usd=5)
    assert plan["start_processed_tokens"]==2_073_161_216
    assert plan["run_until_tokens"]==2_073_292_288
    assert plan["schedule_target_tokens"]==10_000_000_000
    assert plan["gpu_count"]==1
    assert plan["gpu_rented"] is False
    assert plan["training_started"] is False
    assert plan["cost_hard_limit"] is False


@pytest.mark.parametrize("meta,extra", [
    (checkpoint_meta(target=2_000_000_000), 10000),
    (checkpoint_meta(processed=999999,target=10000000), 10000),
    (checkpoint_meta(), 0),
    ({**checkpoint_meta(),"tokenizer_sha256":"0"*64}, 1024),
    ({**checkpoint_meta(),"config":{}},1024),
    ({**checkpoint_meta(),"optimizer":None},1024),
])
def test_invalid_checkpoint_or_schedule_fails_closed(meta,extra):
    with pytest.raises(ValueError):
        make_plan(meta,extra_tokens=extra,price_usd=0.366,cap_usd=5)


def test_4090_hardware_requirements():
    validate_host([{"name":"NVIDIA GeForce RTX 4090","memory_mib":24564}],ram_gb=32)
    with pytest.raises(ValueError,match="exactly one"):
        validate_host([{"name":"RTX 4090","memory_mib":24000}]*2,ram_gb=32)
    with pytest.raises(ValueError):
        validate_host([{"name":"RTX 4070","memory_mib":12000}],ram_gb=32)
    with pytest.raises(ValueError):
        validate_host([{"name":"RTX 4090","memory_mib":24000}],ram_gb=16)


def test_manifest_checks_sha_and_bpe_lineage(tmp_path):
    root=small_data(tmp_path)
    result=validate_data_manifest(root/"manifest.json",seq=512)
    assert result["train_shards"]==1
    assert result["valid_shards"]==1
    path=root/"train"/"train_00000.bin"
    path.write_bytes(b"evil"+path.read_bytes())
    with pytest.raises(ValueError,match="SHA256"):
        validate_data_manifest(root/"manifest.json",seq=512)


def test_preparation_makes_separate_working_copy_without_training(tmp_path):
    root=small_data(tmp_path)
    original=tmp_path/"original-resume.pt";original.write_bytes(b"FAKE-TRUSTED-CHECKPOINT")
    tkn=tmp_path/"tokenizer.json";tkn.write_bytes(b"fake-tokenizer")
    old_hash=sha256_file(original)
    out=tmp_path/"cloud-run"
    def verified(path):
        assert Path(path)==original
        return checkpoint_meta()
    report=prepare_run(
        source_checkpoint=original, checkpoint_sha256=old_hash,
        tokenizer=tkn, tokenizer_sha256=sha256_file(tkn),
        data_manifest=root/"manifest.json", run_directory=out,
        extra_tokens=131072,price_usd=0.366,cap_usd=5,
        checkpoint_loader=verified,
        hardware_check=lambda:validate_host([{"name":"RTX 4090","memory_mib":24564}],32),
        allow_test_tokenizer=True)
    assert report["training_started"] is False
    assert original.read_bytes()==b"FAKE-TRUSTED-CHECKPOINT"
    assert (out/"resume.pt").read_bytes()==original.read_bytes()
    command=(out/"start-vast4090.sh").read_text()
    assert "train350_compiled.py" in command
    assert "--resume" in command
    assert "--compile-mode none" in command
    assert "--stop-after-tokens 2073292288" in command
    assert "python -u" in command
    assert "sudo " not in command
    with pytest.raises(FileExistsError):
        prepare_run(source_checkpoint=original,checkpoint_sha256=old_hash,
                    tokenizer=tkn,tokenizer_sha256=sha256_file(tkn),
                    data_manifest=root/"manifest.json",run_directory=out,
                    extra_tokens=131072,price_usd=0.366,cap_usd=5,
                    checkpoint_loader=verified,
                    hardware_check=lambda:None,allow_test_tokenizer=True)


def test_invalid_checkpoint_digest_prevents_copy(tmp_path):
    root=small_data(tmp_path)
    original=tmp_path/"checkpoint.pt";original.write_bytes(b"X")
    tkn=tmp_path/"tokenizer.json";tkn.write_bytes(b"test")
    out=tmp_path/"run"
    with pytest.raises(ValueError,match="SHA256"):
        prepare_run(source_checkpoint=original,checkpoint_sha256="f"*64,
                    tokenizer=tkn,tokenizer_sha256=sha256_file(tkn),
                    data_manifest=root/"manifest.json",run_directory=out,
                    extra_tokens=131072,price_usd=0.366,cap_usd=5,
                    checkpoint_loader=lambda p:checkpoint_meta(),hardware_check=lambda:None,
                    allow_test_tokenizer=True)
    assert not out.exists()



def test_original_native_trainers_never_resume_with_unrestricted_pickle():
    project=Path(__file__).resolve().parents[1]
    for filename in ("train350.py", "train350_compiled.py"):
        source=(project/"frontier350"/filename).read_text(encoding="utf8")
        assert "from secure_checkpoint350 import load_checkpoint" in source
        assert "ck = load_checkpoint(args.resume)" in source
        assert "weights_only=False" not in source


def test_allowlisted_legacy_rng_checkpoint_can_still_be_loaded(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from frontier350.secure_checkpoint350 import load_checkpoint
    path=tmp_path/"trusted-resume.pt"
    source={
        **checkpoint_meta(),
        "model":{"some_tensor":torch.ones((3,4))},
        "optimizer":{"state":{},"param_groups":[{"params": []}]},
        "torch_rng":torch.get_rng_state(),
        "python_rng":__import__("random").getstate(),
        "numpy_rng":np.random.get_state(),
        "cuda_rng":[],
    }
    torch.save(source,path)
    loaded=load_checkpoint(path)
    assert loaded["processed_tokens"]==source["processed_tokens"]
    assert loaded["target_tokens"]==source["target_tokens"]
    assert torch.equal(loaded["model"]["some_tensor"],source["model"]["some_tensor"])
    assert loaded["numpy_rng"][0]==source["numpy_rng"][0]
    assert np.array_equal(loaded["numpy_rng"][1],source["numpy_rng"][1])
    assert loaded["optimizer"]["param_groups"]==source["optimizer"]["param_groups"]
