"""Local-only CPU speech adapter for NERAL. No model auto-download or cloud API.

Whisper and Piper are optional extras, loaded ONLY from explicitly installed
local_state/voice files; the preview continues to work without either extra.
Audio is processed in RAM, no user recordings or texts are persisted.
"""
from __future__ import annotations

from array import array
from io import BytesIO
import importlib.util
from pathlib import Path
import sys
import threading
import wave

from flask import jsonify, request, send_file

STT_HZ = 16000
MAX_WAV_BYTES = 500_000
MAX_SPEECH_SECONDS = 12
MAX_SPEECH_TEXT = 280
ROOT = Path(__file__).resolve().parent / "local_state" / "voice"
STT_DIR = ROOT / "stt"
TTS_MODEL = ROOT / "tts" / "de_DE-thorsten-medium.onnx"


def validate_wav(blob: bytes) -> array:
    """Parse ONLY mono PCM16 at 16 kHz, between 0.25s and 12s."""
    if not isinstance(blob, bytes) or not 8000 <= len(blob) <= MAX_WAV_BYTES:
        raise ValueError("Unsupported audio size.")
    if not blob.startswith(b"RIFF") or blob[8:12] != b"WAVE":
        raise ValueError("Expected RIFF/WAVE PCM.")
    try:
        with wave.open(BytesIO(blob), "rb") as stream:
            if (stream.getcomptype() != "NONE" or
                    stream.getnchannels() != 1 or
                    stream.getsampwidth() != 2 or
                    stream.getframerate() != STT_HZ):
                raise ValueError("Expected 16-kHz mono PCM16.")
            frames = stream.getnframes()
            if not STT_HZ // 4 <= frames <= STT_HZ * MAX_SPEECH_SECONDS:
                raise ValueError("Audio duration out of range.")
            pcm = stream.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("Incomplete audio.")
    except (EOFError, wave.Error, OSError) as exc:
        raise ValueError("Invalid WAV audio.") from exc
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


class LocalVoiceService:
    """Lazy CPU inference; configured with fixed, non-symlink local files."""
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.stt_dir = self.root / "stt"
        self.tts_model = self.root / "tts" / TTS_MODEL.name
        self._stt_model = None
        self._tts_voice = None
        self._lock = threading.Lock()

    def _stt_available(self) -> bool:
        return bool(importlib.util.find_spec("faster_whisper") is not None
                    and self.stt_dir.is_dir() and not self.stt_dir.is_symlink()
                    and all((self.stt_dir / name).is_file() and
                            not (self.stt_dir / name).is_symlink()
                            for name in ("model.bin","config.json","tokenizer.json")))

    def _tts_available(self) -> bool:
        return bool(importlib.util.find_spec("piper") is not None
                    and self.tts_model.is_file()
                    and not self.tts_model.is_symlink()
                    and self.tts_model.with_suffix(".onnx.json").is_file()
                    and not self.tts_model.with_suffix(".onnx.json").is_symlink())

    def status(self) -> dict:
        # The one-file Windows EXE bundles the Flask preview, NOT optional
        # faster-whisper/Piper native libraries or the voice installer.
        # Files added beside NERAL.exe cannot extend its frozen interpreter.
        if getattr(sys, "frozen", False):
            return {"ready":False,"stt_ready":False,"tts_ready":False,
                    "execution":"local_cpu","audio_leaves_pc":False,
                    "automatic_download":False,"full_duplex":False,
                    "turn_based":True,"installation":"zip_required",
                    "reason":"NERAL.exe enthält die Offline-Sprachpakete nicht. "
                             "Die entpackte NERAL-ZIP-Version starten und "
                             "Install NERAL Voice.cmd ausführen."}
        stt, tts = self._stt_available(), self._tts_available()
        return {"ready":stt and tts, "stt_ready":stt, "tts_ready":tts,
                "execution":"local_cpu", "audio_leaves_pc":False,
                "automatic_download":False, "full_duplex":False,
                "turn_based":True}

    def transcribe(self, blob: bytes) -> str:
        samples=validate_wav(blob)
        if not self._stt_available():
            raise RuntimeError("Offline STT not installed.")
        if self._stt_model is None:
            from faster_whisper import WhisperModel
            self._stt_model=WhisperModel(
                str(self.stt_dir),device="cpu",compute_type="int8",
                local_files_only=True, cpu_threads=4)
        import numpy as np
        waveform=np.asarray(samples,dtype=np.float32)/32768.0
        segments,_=self._stt_model.transcribe(
            waveform,language="de",beam_size=1,condition_on_previous_text=False,
            without_timestamps=True)
        # Generator must be consumed to complete actual inference.
        result=" ".join(s.text.strip() for s in segments).strip()
        if len(result.encode("utf-8"))>4096:
            raise ValueError("Speech text exceeded chat input limit.")
        return result

    def synthesize(self, text: str) -> bytes:
        if not self._tts_available():
            raise RuntimeError("Offline TTS not installed.")
        if self._tts_voice is None:
            from piper import PiperVoice
            self._tts_voice=PiperVoice.load(str(self.tts_model),use_cuda=False)
        memory=BytesIO()
        with wave.open(memory,"wb") as target:
            self._tts_voice.synthesize_wav(text,target)
        wav=memory.getvalue()
        if not wav.startswith(b"RIFF") or len(wav)>3_000_000:
            raise RuntimeError("Invalid or oversized local speech synthesis.")
        return wav


def register_local_voice_api(app, service=None):
    """All endpoints inherit Flask's localhost, same-origin and CSRF guard."""
    service = LocalVoiceService() if service is None else service
    lock = threading.Lock()

    @app.get("/api/neral/voice/status")
    def local_voice_status():
        return jsonify(service.status())

    @app.post("/api/neral/voice/transcribe")
    def local_voice_transcribe():
        if request.mimetype != "audio/wav":
            return jsonify(error="Nur WAV PCM16 erlaubt."),415
        blob=request.get_data(cache=False)
        try:
            validate_wav(blob)
        except ValueError:
            return jsonify(error="Nur 16-kHz-Mono-WAV von 0,25 bis 12 Sekunden."),400
        if not service.status().get("stt_ready"):
            return jsonify(error="Lokale Spracherkennung fehlt. Offline-Sprachpaket installieren."),503
        if not lock.acquire(blocking=False):
            return jsonify(error="Lokaler Sprachprozessor ist beschäftigt."),409
        try:
            try:
                text=service.transcribe(blob)
                if not isinstance(text,str) or len(text.encode("utf-8"))>4096:
                    raise ValueError("Invalid transcript")
            except Exception:
                return jsonify(error="Lokale Spracherkennung fehlgeschlagen."),503
            return jsonify(text=text,local_only=True,
                           stt_engine="faster_whisper_cpu",stored=False)
        finally:
            lock.release()

    @app.post("/api/neral/voice/speak")
    def local_voice_speak():
        if not request.is_json:
            return jsonify(error="Nur JSON erlaubt."),415
        data=request.get_json(silent=True)
        if (not isinstance(data,dict) or set(data)!={"text"} or
                not isinstance(data["text"],str) or
                not data["text"].strip() or
                len(data["text"])>MAX_SPEECH_TEXT or
                any(ord(ch)<32 and ch not in "\n\t" for ch in data["text"])):
            return jsonify(error="Text für lokale Sprachausgabe ungültig oder zu lang."),400
        if not service.status().get("tts_ready"):
            return jsonify(error="Lokale deutsche Stimme fehlt. Offline-Sprachpaket installieren."),503
        if not lock.acquire(blocking=False):
            return jsonify(error="Lokaler Sprachprozessor ist beschäftigt."),409
        try:
            try:
                speech=service.synthesize(data["text"])
                if (not isinstance(speech,bytes) or
                        not speech.startswith(b"RIFF") or len(speech)>3_000_000):
                    raise ValueError("Invalid audio")
            except Exception:
                return jsonify(error="Lokale Sprachausgabe fehlgeschlagen."),503
            return send_file(BytesIO(speech),mimetype="audio/wav",
                             as_attachment=False,download_name="neral.wav",
                             max_age=0)
        finally:
            lock.release()

    return service
