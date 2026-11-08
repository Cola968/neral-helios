"""Frozen one-file NERAL.exe cannot run optional Piper/Whisper packages.

The published EXE build installs requirements-neral-preview only. It must
never imply offline voice is ready or direct users to an installer absent
from its one-file distribution.
"""
from __future__ import annotations
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from neral_local_voice import LocalVoiceService

ROOT=Path(__file__).resolve().parents[1]


class OneClickVoiceStatusTests(unittest.TestCase):
    def test_onefile_exe_reports_unsupported_offline_voice_instead_of_fake_ready(self):
        with patch.object(sys,"frozen",True,create=True):
            status=LocalVoiceService().status()
        self.assertFalse(status["ready"])
        self.assertFalse(status["stt_ready"])
        self.assertFalse(status["tts_ready"])
        self.assertEqual(status["installation"],"zip_required")
        self.assertIn("NERAL.exe",status["reason"])
        self.assertIn("ZIP",status["reason"])
        self.assertFalse(status["audio_leaves_pc"])

    def test_readme_explains_which_distribution_supports_voice_installer(self):
        entry=(ROOT/"START HIER.txt").read_text("utf8")
        self.assertIn("NERAL.exe",entry)
        self.assertIn("Install NERAL Voice.cmd",entry)
        self.assertIn("ZIP",entry)

    def test_exe_build_still_has_no_unreviewed_speech_cloud_or_native_dependencies(self):
        workflow=(ROOT/".github/workflows/neral-single-click-windows.yml").read_text("utf8")
        self.assertIn("requirements-neral-preview.txt pyinstaller",workflow)
        self.assertNotIn("pip install --upgrade piper-tts",workflow)


if __name__=="__main__":
    unittest.main()
