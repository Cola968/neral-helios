"""Offline speech contracts: fail closed, no cloud STT/TTS or model storage."""
from __future__ import annotations
import io
import math
import struct
import unittest
import wave
from unittest.mock import patch

from neral_preview_server import create_preview_app
from neral_local_voice import LocalVoiceService, validate_wav

HOST = "http://127.0.0.1:9690"


def audio(seconds=1.0, rate=16000, channels=1, width=2):
    samples=int(seconds*rate)
    with io.BytesIO() as stream:
        with wave.open(stream,"wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(width)
            writer.setframerate(rate)
            writer.writeframes(b"\0"*(samples*channels*width))
        return stream.getvalue()


class FakeModel:
    model="qwen3:8b"
    def respond_chat(self, messages):
        return "Eine Antwort."


class FakeLocalSpeech:
    def __init__(self):
        self.seen=[]
    def status(self):
        return {"ready":True,"stt_ready":True,"tts_ready":True,
                "execution":"local_cpu","audio_leaves_pc":False,
                "automatic_download":False}
    def transcribe(self, data):
        self.seen.append(("stt",data))
        return "Hallo NERAL"
    def synthesize(self, text):
        self.seen.append(("tts",text))
        return audio(0.7)


class OfflineVoiceServerTests(unittest.TestCase):
    def setUp(self):
        self.speech=FakeLocalSpeech()
        self.client=create_preview_app(FakeModel(),secret_key=b"X"*32,
                                       voice_service=self.speech).test_client()
        self.assertEqual(self.client.get("/",base_url=HOST).status_code,200)
        res=self.client.get("/api/neral/preview/session",base_url=HOST)
        self.h={"X-Neral-CSRF":res.json["csrf"]}

    def test_status_readiness_has_explicit_offline_provenance(self):
        r=self.client.get("/api/neral/voice/status",base_url=HOST)
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json["execution"],"local_cpu")
        self.assertIs(r.json["audio_leaves_pc"],False)
        self.assertIs(r.json["automatic_download"],False)

    def test_valid_wav_local_transcript_no_upload_or_remote_service(self):
        data=audio(0.7)
        r=self.client.post("/api/neral/voice/transcribe",data=data,
            headers={**self.h,"Content-Type":"audio/wav"},base_url=HOST)
        self.assertEqual(r.status_code,200,r.json)
        self.assertEqual(r.json["text"],"Hallo NERAL")
        self.assertIs(r.json["local_only"],True)
        self.assertEqual(self.speech.seen,[("stt",data)])

    def test_tts_is_real_wav_bytes_local_with_sanitized_text(self):
        r=self.client.post("/api/neral/voice/speak",
            json={"text":"Guten Tag."},headers=self.h,base_url=HOST)
        self.assertEqual(r.status_code,200,r.json if r.is_json else "")
        self.assertEqual(r.mimetype,"audio/wav")
        self.assertTrue(r.data.startswith(b"RIFF"))
        self.assertEqual(self.speech.seen[0],("tts","Guten Tag."))

    def test_cross_site_and_csrf_fail_before_audio_engine(self):
        for h in ({}, {**self.h,"Origin":"https://evil.invalid"}):
            r=self.client.post("/api/neral/voice/transcribe",
                data=audio(),headers={**h,"Content-Type":"audio/wav"},
                base_url=HOST)
            self.assertEqual(r.status_code,403)
        self.assertFalse(self.speech.seen)

    def test_wav_rejects_invalid_sizes_codec_and_duration_before_model(self):
        samples=[b"NO",audio(1,rate=8000),audio(1,channels=2),
                 audio(1,width=1),audio(0.1),audio(16)]
        for bad in samples:
            with self.subTest(length=len(bad)):
                r=self.client.post("/api/neral/voice/transcribe",
                    data=bad,headers={**self.h,"Content-Type":"audio/wav"},
                    base_url=HOST)
                self.assertIn(r.status_code,(400,413))
        self.assertFalse(self.speech.seen)

    def test_refuse_server_stt_tts_before_model_install(self):
        svc=LocalVoiceService()
        with patch.object(svc,"_stt_available",return_value=False),patch.object(
                 svc,"_tts_available",return_value=False):
            client=create_preview_app(FakeModel(),secret_key=b"Y"*32,
                                      voice_service=svc).test_client()
            client.get("/",base_url=HOST)
            s=client.get("/api/neral/preview/session",base_url=HOST)
            h={"X-Neral-CSRF":s.json["csrf"]}
            self.assertIs(client.get("/api/neral/voice/status",
                                     base_url=HOST).json["ready"],False)
            x=client.post("/api/neral/voice/transcribe",data=audio(),
                  headers={**h,"Content-Type":"audio/wav"},base_url=HOST)
            self.assertEqual(x.status_code,503)

    def test_invalid_tts_payload_and_large_text_never_synthesized(self):
        for item in ({}, {"text":""}, {"text":"x"*1601},
                     {"text":"Test","provider":"remote"}):
            r=self.client.post("/api/neral/voice/speak",json=item,
                               headers=self.h,base_url=HOST)
            self.assertEqual(r.status_code,400)
        self.assertFalse(self.speech.seen)

    def test_local_wav_decode_is_bounded_and_fully_pcm16(self):
        result=validate_wav(audio())
        self.assertEqual(len(result),16000)
        self.assertEqual(result[0],0)
        for data in (b"fake",audio(1,rate=44100),audio(1,channels=2),
                     audio(0.1),audio(16)):
            with self.assertRaises(ValueError):
                validate_wav(data)


if __name__=="__main__":
    unittest.main()
