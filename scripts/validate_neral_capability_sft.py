"""Validate reviewed SFT/DEV data, optionally with a verified local tokenizer.

No trainer, model-weight loader, download, resealing, dataset mutation or uploads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neral_control import corpus


def validate_data(root, *, tokenizer_dir=None):
    evidence=corpus.validate(Path(root))
    tokenizer_provenance=None
    if tokenizer_dir is not None:
        # Verify the pinned safetensors/tokenizer inventory before importing the
        # tokenizer library. Reading hashes does not deserialize/load weights.
        from model_catalog import validate_hf_snapshot
        snapshot=validate_hf_snapshot(Path(tokenizer_dir).absolute(),verify_hashes=True)
        from transformers import Qwen2TokenizerFast
        from qwen_local_runtime import CHAT_TEMPLATE
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(tokenizer_dir),
            local_files_only=True,trust_remote_code=False,token=False)
        evidence=corpus.validate(Path(root),tokenizer)
        tokenizer_provenance=dict(base_model=snapshot['base_model'],revision=snapshot['revision'],
            snapshot_sha256=snapshot['sha256'],
            chat_template_sha256=hashlib.sha256(CHAT_TEMPLATE.encode('utf8')).hexdigest())
    return {**evidence,'stage':'DATA_VALIDATION_ONLY','tokenizer_provenance':tokenizer_provenance,
            'model_weights_loaded':False,'training_executed':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tokenizer-dir',type=Path)
    args=parser.parse_args(argv)
    try:result=validate_data(args.root,tokenizer_dir=args.tokenizer_dir)
    except (ValueError,OSError,KeyError,TypeError,ImportError) as exc:
        parser.exit(2,f'Data validation refused: {type(exc).__name__}; no training executed.\n')
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
