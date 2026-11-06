"""Curriculum quality gates, no GPU, no network, no trainer side effects."""
import json
import shutil
from pathlib import Path
import pytest

from scripts.validate_neral_curriculum import validate, audit_answer

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets"/"neral_curriculum_v1"


def test_original_corpus_contains_1000_verified_separated_examples():
    summary=validate(DATA)
    assert summary["valid"]
    assert summary["total_rows"]==1000
    assert summary["unique_prompts"]==1000
    assert summary["split_stats"]["train"]["rows"]==800
    assert summary["split_stats"]["val"]["rows"]==100
    assert summary["split_stats"]["test"]["rows"]==100
    assert summary["tokenizer_validated"] is False
    assert summary["training_performed"] is False


def test_altered_answer_detected_without_model():
    with pytest.raises((ValueError,AssertionError)):
        audit_answer("physics","Physik: Masse 10 kg, konstante Beschleunigung 3 m/s^2.",
                     "F = 1000 N")


def test_unapproved_training_not_misrepresented():
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_allowed"] is False
    assert manifest["quality_improvement_proven"] is False
    assert "train" in manifest["splits"] and "test" in manifest["splits"]
    # No private user prompts, credentials or external data sources.
    assert manifest["source"].startswith("original synthetic")


def test_duplicate_prompt_in_eval_split_rejected(tmp_path):
    for name in ("manifest.json","train.jsonl","val.jsonl","test.jsonl"):
        shutil.copyfile(DATA/name,tmp_path/name)
    train=(tmp_path/"train.jsonl").read_text().splitlines()
    test=(tmp_path/"test.jsonl").read_text().splitlines()
    item=json.loads(test[0]); item["messages"][1]["content"]=json.loads(train[0])["messages"][1]["content"]
    test[0]=json.dumps(item)
    (tmp_path/"test.jsonl").write_text("\n".join(test)+"\n")
    with pytest.raises(ValueError,match="test"):
        validate(tmp_path)


def test_untrusted_sql_and_broken_python_labels_rejected():
    with pytest.raises(ValueError):
        audit_answer("sql_group","SQL: Tabelle Track(TrackID,GenreID). Nur Gruppen mit GenreID >= 100 beruecksichtigen. Liste GenreID-Gruppen mit mindestens 3 Zeilen, nach GenreID aufsteigend. Gib nur SQL aus.",
                     "DROP TABLE Track;")
    with pytest.raises((ValueError,SyntaxError)):
        audit_answer("python_debug","Korrigiere diesen Code:\ndef solve_3(x):",
                     "def solve_3(:")
