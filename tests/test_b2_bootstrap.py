import json
from pathlib import Path

from bootstrap_b2_sources import SOURCE_SPECS, make_manifest, safe_id


def test_safe_id_is_stable_and_sanitizes_bad_values():
    assert safe_id("doc-17",5)=="doc-17"
    assert safe_id("https://example.com/a",5)=="https://example.com/a"
    assert safe_id("bad id with spaces",5)=="row-000000005"
    assert safe_id(None,9)=="row-000000009"


def test_bootstrap_source_mix_is_exact_and_bilingual():
    assert abs(sum(s["target_fraction"] for s in SOURCE_SPECS)+.10-1.0)<1e-12
    assert {s["language"] for s in SOURCE_SPECS}=={"de","en"}
    assert any(s["domain"]=="education" for s in SOURCE_SPECS)


def test_manifest_records_exact_local_hashes(tmp_path: Path):
    reports={}
    for spec in SOURCE_SPECS:
        path=tmp_path/f"{spec['id']}.jsonl";path.write_text('{"id":"x","text":"placeholder"}\n',encoding="utf8")
        import hashlib
        reports[spec["id"]]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"text_bytes":1000,"documents":1}
    d=tmp_path/"oasst1-dialogue.jsonl";d.write_text('{"id":"x","text":"placeholder"}\n',encoding="utf8")
    import hashlib
    reports["oasst1-dialogue"]={"sha256":hashlib.sha256(d.read_bytes()).hexdigest(),"text_bytes":1000,"documents":1,"upstream_sha256":"a"*64}
    manifest=make_manifest(tmp_path,reports)
    assert len(manifest["sources"])==4
    assert abs(sum(s["target_fraction"] for s in manifest["sources"])-1.0)<1e-12
    loaded=json.loads((tmp_path/"manifest.json").read_text(encoding="utf8"))
    assert loaded["sources"][0]["sha256"]==reports[SOURCE_SPECS[0]["id"]]["sha256"]
