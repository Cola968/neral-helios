"""Offline NERAL voice must be actually usable after installer reports success.

Current prior installation only tested file presence, not real STT or TTS
inference. Those fake-ready states caused a clickable but nonworking mic.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import Mock
import wave

ROOT = Path(__file__).resolve().parents[1]


def valid_tts_wav() -> bytes:
    stream = BytesIO()
    with wave.open(stream, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x01" * 5000)
    return stream.getvalue()


class LocalVoiceInstallTests(unittest.TestCase):
    def test_piper_runtime_is_pinned_to_known_self_contained_windows_release(self):
        requirements = (ROOT / "requirements-neral-local-voice.txt").read_text("utf8")
        self.assertIn("piper-tts==1.3.0", requirements)
        self.assertNotIn("piper-tts>=1.3", requirements)

    def test_optional_installer_runs_real_stt_and_tts_check_before_success(self):
        script = (ROOT / "Install-Neral-Local-Voice.ps1").read_text("utf8")
        self.assertIn("scripts\\neral_local_voice_selftest.py", script)
        self.assertIn("if($LASTEXITCODE -ne 0)", script)
        self.assertIn("Read-Host", script)
        self.assertNotIn("pip install --upgrade piper-tts", script)

    def test_user_start_instructions_explain_voice_is_not_preinstalled(self):
        start = (ROOT / "START HIER.txt").read_text("utf8")
        self.assertIn("Install NERAL Voice.cmd", start)
        self.assertIn("Lokaler Sprachmodus", start)
        self.assertIn("neu starten", start.lower())

    def test_cpu_speech_selftest_exercises_real_synthesis_and_transcription(self):
        from scripts.neral_local_voice_selftest import check_service
        service = Mock()
        service.status.return_value = {"stt_ready":True,"tts_ready":True}
        service.transcribe.return_value = ""
        service.synthesize.return_value = valid_tts_wav()
        result = check_service(service)
        self.assertTrue(result["ready"])
        service.transcribe.assert_called_once()
        supplied = service.transcribe.call_args.args[0]
        self.assertTrue(supplied.startswith(b"RIFF"))
        service.synthesize.assert_called_once_with("Hallo NERAL.")
        self.assertGreater(result["tts_frames"], 0)

    def test_cpu_speech_selftest_refuses_fake_ready_engine(self):
        from scripts.neral_local_voice_selftest import check_service
        engine = Mock()
        engine.status.return_value = {"stt_ready":False,"tts_ready":True}
        with self.assertRaises(RuntimeError):
            check_service(engine)
        engine.synthesize.assert_not_called()

    def test_cpu_speech_selftest_refuses_invalid_piper_wav(self):
        from scripts.neral_local_voice_selftest import check_service
        engine = Mock()
        engine.status.return_value = {"stt_ready":True,"tts_ready":True}
        engine.transcribe.return_value = ""
        engine.synthesize.return_value = b"RIFF_CORRUPTED"
        with self.assertRaises((RuntimeError, EOFError, wave.Error)):
            check_service(engine)

    def test_installed_speech_file_change_forces_fresh_release(self):
        from scripts.build_neral_1_0_preview import REQUIRED_FILES
        self.assertIn("scripts/neral_local_voice_selftest.py", REQUIRED_FILES)
        workflow = (ROOT / ".github/workflows/neral-1-0-release-rc.yml").read_text("utf8")
        self.assertGreaterEqual(workflow.count("'scripts/neral_local_voice_selftest.py'"), 2)


if __name__ == "__main__":
    unittest.main()
