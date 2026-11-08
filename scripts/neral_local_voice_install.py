"""Explicitly invoked one-time download of open CPU voice models.

Never imported by the chat server. The ordinary NERAL startup does NOT call
this tool. Run only after the Windows installer has asked user permission.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

STT_REPO="Systran/faster-whisper-base"
STT_REV="ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
TTS_REPO="rhasspy/piper-voices"
TTS_REV="v1.0.0"
TTS_NAME="de_DE-thorsten-medium.onnx"
TTS_PREFIX="de/de_DE/thorsten/medium/"
# Independently published model digests, verified after opt-in download.
STT_SHA="d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9"
TTS_SHA="7e64762d8e5118bb578f2eea6207e1a35a8e0c30595010b666f983fc87bb7819"


def checksum(path):
    sha=hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024*1024),b""):
            sha.update(block)
    return sha.hexdigest()


def install(root: Path):
    from huggingface_hub import hf_hub_download, snapshot_download
    root=Path(root).resolve()
    target=root/"stt"
    target.mkdir(parents=True,exist_ok=True)
    snapshot_download(
        repo_id=STT_REPO,revision=STT_REV,local_dir=str(target),
        allow_patterns=["model.bin","config.json","tokenizer.json",
                        "preprocessor_config.json","vocabulary.txt"],
    )
    if checksum(target/"model.bin")!=STT_SHA:
        raise RuntimeError("Offline STT model digest mismatch.")
    tts=root/"tts"
    tts.mkdir(parents=True,exist_ok=True)
    for name in (TTS_NAME,TTS_NAME+".json"):
        source=Path(hf_hub_download(
            repo_id=TTS_REPO,revision=TTS_REV,
            filename=TTS_PREFIX+name,
            local_dir=str(root/"voice_downloads")))
        destination=tts/name
        if destination.exists() and destination.is_symlink():
            raise RuntimeError("Refusing symlink voice asset.")
        shutil.copyfile(source,destination)
    if checksum(tts/TTS_NAME)!=TTS_SHA:
        raise RuntimeError("Offline German TTS model digest mismatch.")
    print("NERAL offline voice assets installed to local_state/voice.")
    print("Runtime uses CPU-only local model paths, no external audio service.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    install(args.root)


if __name__=="__main__":
    main()
