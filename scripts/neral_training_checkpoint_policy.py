"""Conservative opt-in validation checkpoint selection for future local QLoRA.

No training or GPU operations here. Historical v6/v61 TrainingArguments
are unchanged by default. Research experiments may select the adapter from
the *best held-out validation loss* checkpoint, never automatically promote.
"""
from __future__ import annotations

import math
from pathlib import Path
import re


def options_for_training(*,steps:int,interval:int|None,
                         max_steps_limit:int=120)->dict:
    if type(max_steps_limit) is not int or max_steps_limit not in (120,240):
        raise ValueError("Only reviewed 120- or 240-step manual policies")
    if type(steps) is not int or not 1<=steps<=max_steps_limit:
        raise ValueError("Only bounded, manual experimental training schedules")
    if interval is None:
        return {"eval_strategy":"no","save_strategy":"no"}
    if steps<30 or (type(interval) is not int or
            interval<10 or interval>steps//3 or steps%interval):
        raise ValueError("Only aligned, >=3-evaluation checkpoint schedules")
    return {
        "eval_strategy":"steps","eval_steps":interval,
        "save_strategy":"steps","save_steps":interval,
        "load_best_model_at_end":True,
        "metric_for_best_model":"eval_loss",
        "greater_is_better":False,
        "save_total_limit":2,
    }


def validate_best_checkpoint(path:str|Path,loss:float,root:Path,
                             *,steps:int,interval:int,
                             max_steps_limit:int=120)->int:
    """Return verified private best step, never return public weight paths."""
    options_for_training(steps=steps,interval=interval,
                         max_steps_limit=max_steps_limit)
    p,root=Path(path),Path(root)
    if (root.is_symlink() or not root.is_dir() or
            p.is_symlink() or not p.is_dir() or
            p.parent.resolve()!=root.resolve()):
        raise ValueError("Checkpoint must be real and under private Trainer root")
    m=re.fullmatch(r"checkpoint-([0-9]{1,5})",p.name)
    if not m or (step:=int(m.group(1))) not in range(interval,steps+1,interval):
        raise ValueError("Unrecognized or misaligned selected checkpoint")
    if type(loss) not in (int,float) or not math.isfinite(float(loss)) or loss<0:
        raise ValueError("Missing or nonfinite held-out validation loss")
    return step
