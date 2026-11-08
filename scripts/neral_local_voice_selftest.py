"""Offline voice READY means the real CPU engines worked, not just files exist.

Executed only after the user explicitly approves downloading models/packages.
Uses synthetic silence and a fixed sentence; no real microphone, user content,
network call, personal data, or disk recording.
"""
from __future__ import annotations

from io import BytesIO
import wave


def synthetic_silence():
    out=BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0" * 32000)
    return out.getvalue()


def check_service(service):
    status=service.status()
    if status.get("stt_ready") is not True:
        raise RuntimeError("Faster-Whisper nicht installiert oder Modell unvollständig.")
    if status.get("tts_ready") is not True:
        raise RuntimeError("Piper nicht installiert oder deutsche Stimme unvollständig.")

    # Consume actual local Whisper inference. A silent recording may produce
    # empty text; that is not a failure. The test verifies execution, not words.
    transcript=service.transcribe(synthetic_silence())
    if not isinstance(transcript, str):
        raise RuntimeError("Faster-Whisper lieferte keinen Text.")

    wav=service.synthesize("Hallo NERAL.")
    if not isinstance(wav, bytes) or not wav.startswith(b"RIFF"):
        raise RuntimeError("Piper lieferte kein WAV-Audio.")
    try:
        with wave.open(BytesIO(wav), "rb") as output:
            frames=output.getnframes()
            channels=output.getnchannels()
            width=output.getsampwidth()
            rate=output.getframerate()
            if (frames < 1000 or channels != 1 or
                    width != 2 or not 8000 <= rate <= 48000):
                raise RuntimeError("Piper erzeugte ungültige oder leere Sprache.")
    except (EOFError, wave.Error) as exc:
        raise RuntimeError("Piper erzeugte beschädigte WAV-Daten.") from exc

    return {"ready":True,"stt":"faster-whisper cpu int8",
            "tts":"piper cpu", "tts_frames":frames,
            "audio_leaves_pc":False}


def main():
    from neral_local_voice import LocalVoiceService
    try:
        result=check_service(LocalVoiceService())
    except Exception as exc:
        # Installer needs a meaningful failure, never a misleading READY.
        # Avoid printing local paths or model/user audio.
        print("NERAL Offline Voice: NICHT BEREIT.",
              "Whisper/Piper-Inferenz konnte nicht abgeschlossen werden.")
        print("Fehlerart:", type(exc).__name__)
        raise SystemExit(1)
    print("NERAL Offline Voice: BEREIT.",
          "Echte CPU-STT- und TTS-Inferenz bestanden.",
          "Piper WAV-Frames:",result["tts_frames"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
