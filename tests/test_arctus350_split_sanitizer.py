"""Copy approved NeralTalk splits without exact later-turn leakage."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from frontier350.chat350_split_sanitizer import sanitize_splits, write_sanitized_dataset
from frontier350.chat350_data import validate_dataset_root,validate_prompt_holdouts


def row(key, *messages):
    return {"id":key,"language":"de","domain":"general","source":"synthetic","license":"CC0",
            "messages":[{"role":"user" if i%2==0 else "assistant","content":text}
                        for i,text in enumerate(messages)]}


class SanitizationTests(unittest.TestCase):
    def test_drops_train_contamination_but_keeps_test_val(self):
        data={
            "train":[row("tr1","Hello","Hi","Gemeinsame spätere Frage?","Antwort"),
                     row("tr2","Nur im Training","okay")],
            "val":[row("val1","Ganz andere Frage","okay","Gemeinsame spätere Frage?","Anders")],
            "test":[row("test1","Unabhängiger Test","okay")],
        }
        clean,counts=sanitize_splits(data)
        self.assertEqual([r["id"] for r in clean["train"]],["tr2"])
        self.assertEqual(counts["dropped"]["train"],1)
        validate_prompt_holdouts(clean)

    def test_test_preserved_val_collision_removed(self):
        clean,counts=sanitize_splits({
          "train":[row("tr","Anders als alle","okay")],
          "val":[row("v1","Eine Frage","okay","Wiederkehrend","Ja"),
                 row("v2","Eigene Val","okay")],
          "test":[row("te","Test Original","okay","Wiederkehrend","Nein")],
        })
        self.assertEqual([r["id"] for r in clean["val"]],["v2"])
        self.assertEqual([r["id"] for r in clean["test"]],["te"])
        self.assertEqual(counts["dropped"]["val"],1)
        validate_prompt_holdouts(clean)

    def test_numeric_templates_not_automatically_deleted(self):
        clean,counts=sanitize_splits({
          "train":[row("tr","Was ist 7 plus 8?","15")],
          "val":[row("va","Was ist 6 plus 8?","14")],
          "test":[row("te","Was ist 5 plus 8?","13")],
        })
        self.assertEqual(counts["dropped"],{"train":0,"val":0,"test":0})

    def test_refuse_empty_required_split(self):
        with self.assertRaises(ValueError):
            sanitize_splits({
                "train":[row("tr","Gleich","A")],
                "val":[row("val","Gleich","B")],
                "test":[row("te","Test","C")],
            })

    def test_real_file_workflow_never_overwrites_source_or_output(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/"original"
            src.mkdir()
            data={
                "train":[row("tr1","Erste Frage","okay","Leck","Beispiel"),
                         row("tr2","Einzigartig Train","okay")],
                "val":[row("va","Validieren","okay","Leck","Anders")],
                "test":[row("te","Test","okay")],
            }
            for split,rows in data.items():
                (src/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
            original={s:(src/f"{s}.jsonl").read_bytes() for s in data}
            dest=Path(td)/"sanitized"
            rep=write_sanitized_dataset(src,dest)
            self.assertEqual(rep["dropped"]["train"],1)
            self.assertEqual((src/"train.jsonl").read_bytes(),original["train"])
            validate_prompt_holdouts(validate_dataset_root(dest)[1])
            with self.assertRaises(FileExistsError):
                write_sanitized_dataset(src,dest)
            with self.assertRaises(ValueError):
                write_sanitized_dataset(src,src)
            serialized=json.dumps(rep)
            for secret in ("Einzigartig Train","Leck","tr1","Validieren"):
                self.assertNotIn(secret,serialized)


if __name__=="__main__":
    unittest.main()
