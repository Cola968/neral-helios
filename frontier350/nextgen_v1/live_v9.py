"""Research-only V9 launcher: print every ACTUAL completed optimizer-step log.

All training arguments pass unmodified to train_guarded_v9.main(). This wrapper
is not a different optimizer or a speed improvement. Lines are JSON and flush
immediately so the GitHub Actions job displays them while the process runs.
"""
import json
import train_guarded_v9 as trainer

_original_append_jsonl = trainer.append_jsonl


def append_and_stream(path, row):
    _original_append_jsonl(path, row)
    if row.get("event") == "train":
        print(json.dumps({"stream": "nextgen_real_completed_optimizer_step", **row}, sort_keys=True), flush=True)


if __name__ == "__main__":
    trainer.append_jsonl = append_and_stream
    trainer.main()
