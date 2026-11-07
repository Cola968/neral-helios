"""CPU contract for read-only real Helios 7B LoRA smoke on Windows."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_v5_helios_smoke import make_report, write_report


class FakePeft:
    peft_config={"default":object()}

class FakeBackend:
    adapter_sha256="a"*64
    _model=FakePeft()
    model="NERAL-local-r16-experimental"


class V5HeliosLiveSmokeContract(unittest.TestCase):
    def test_proof_contains_only_safe_scalars(self):
        raw="Lokale Antwort, bleibt privat."
        r=make_report(FakeBackend(),raw,12.5)
        self.assertEqual(r["status"],"NERAL_V5_HELIOS_LOCAL_ADAPTER_LOADED")
        self.assertEqual(r["base_revision"],"c03e6d358207e414f1eca0bb1891e29f1db0e242")
        self.assertEqual(r["adapter_sha256"],"a"*64)
        self.assertEqual(r["response_length"],len(raw))
        self.assertTrue(r["adapter_loaded"])
        self.assertFalse(r["release_validated"])
        self.assertNotIn(raw,json.dumps(r,ensure_ascii=False))
        self.assertNotIn("prompt",json.dumps(r))
        self.assertNotIn("local_raw",json.dumps(r))
        self.assertNotIn("generated_text",json.dumps(r))

    def test_cannot_claim_loaded_without_peft_model(self):
        backend=FakeBackend()
        backend._model=object()
        with self.assertRaises(ValueError):
            make_report(backend,"Antwort",3)
        backend._model=None
        with self.assertRaises(ValueError):
            make_report(backend,"Antwort",3)

    def test_rejects_empty_overlarge_and_unbounded_duration(self):
        for raw,duration in (("",1),(" "*5,1),("a"*3000,1),("ok",-3),("ok",3600*2)):
            with self.subTest(raw=raw[:30],duration=duration):
                with self.assertRaises(ValueError):
                    make_report(FakeBackend(),raw,duration)

    def test_no_overwrite_and_private_log_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"status.json"
            write_report(path,make_report(FakeBackend(),"LOCAL",10))
            data=json.loads(path.read_text())
            self.assertEqual(data["response_length"],5)
            self.assertNotIn("LOCAL",path.read_text())
            with self.assertRaises(ValueError):
                write_report(path,make_report(FakeBackend(),"LOCAL",10))

    def test_workflow_uses_single_explicit_v5_adapter_and_no_training(self):
        workflow=Path(".github/workflows/neral-v5-helios-smoke.yml").read_text(encoding="utf8")
        for needle in (
            "refs/heads/main","self-hosted, windows, x64, neral-gpu",
            "neral-qwen7b-r32-gpu-exclusive",
            "v5-r16-480-42378021782-1","scripts/neral_v5_helios_smoke.py",
            "SMOKE_NERAL_V5_HELIOS_LOCAL",
            "upload-artifact",
            "if-no-files-found: error",
        ):
            self.assertIn(needle,workflow)
        for unsafe in ("trainer.train(", "train.py", "model.generate(", "git push", "Invoke-WebRequest"):
            self.assertNotIn(unsafe,workflow)


if __name__ == "__main__":
    unittest.main()
