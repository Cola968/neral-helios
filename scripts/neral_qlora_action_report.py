"""Allowlisted scalar-only reporting for the Qwen7B LoRA rank32 three-step pilot.

Never serializes model weights, prompts, datasets, raw trainer logs or paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_STEPS = 3
EXPECTED_TRAINABLE = 80_740_352
ALLOWED_PHASES = {'STARTED', 'PREFLIGHT_PASSED', 'TRAINED', 'REFUSED'}


class ReportError(ValueError):
    pass


def _float(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ReportError('Invalid nonfinite scalar telemetry')
    return float(value)


def _nonnegative(value):
    value = _float(value)
    if value < 0:
        raise ReportError('Negative telemetry')
    return value


def sha_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
        raise ReportError('Missing, linked, or too-large adapter')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize_run(report: dict, adapter_file: Path) -> dict:
    if not isinstance(report, dict) or report.get('status') != 'completed':
        raise ReportError('Training not completed')
    if type(report.get('steps')) is not int or report['steps'] != EXPECTED_STEPS:
        raise ReportError('Not exactly 3 optimizer steps')
    if type(report.get('neral_trainable_parameters')) is not int or report['neral_trainable_parameters'] != EXPECTED_TRAINABLE:
        raise ReportError('Trainable LoRA parameter count not certified')
    metrics = report.get('metrics')
    if not isinstance(metrics, dict) or type(report.get('optimizer_steps_completed')) is not int or report['optimizer_steps_completed'] != EXPECTED_STEPS:
        raise ReportError('Trainer did not report exactly 3 steps')
    logs = report.get('logs')
    times = report.get('optimizer_step_seconds')
    if not isinstance(logs, list) or not isinstance(times, list) or len(times) != EXPECTED_STEPS:
        raise ReportError('Missing step telemetry')
    losses = {}
    for row in logs:
        if not isinstance(row, dict):
            raise ReportError('Invalid log row')
        if 'loss' in row:
            step = row.get('step')
            if type(step) is not int or not 1 <= step <= EXPECTED_STEPS or step in losses:
                raise ReportError('Invalid or repeated loss step')
            losses[step] = _nonnegative(row['loss'])
    if set(losses) != set(range(1, EXPECTED_STEPS + 1)):
        raise ReportError('Not all 3 optimizer-step losses recorded')
    steps = []
    for i, duration in enumerate(times, 1):
        steps.append({'optimizer_step': i, 'duration_seconds': _nonnegative(duration), 'loss': losses[i]})
    out = {
        'schema': 1,
        'status': 'TRAINED_NOT_DEPLOYED',
        'model': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'adapter_kind': 'NERAL LoRA r32',
        'optimizer_steps': 3,
        'trainable_adapter_parameters': EXPECTED_TRAINABLE,
        'step_metrics': steps,
        'train_loss': _nonnegative(metrics['train_loss']),
        'elapsed_seconds': _nonnegative(report['elapsed_seconds']),
        'gpu_peak_allocated_bytes': int(_nonnegative(report['peak_allocated_bytes'])),
        'gpu_peak_reserved_bytes': int(_nonnegative(report['peak_reserved_bytes'])),
        'gpu_peak_sampled_used_bytes': int(_nonnegative(report.get('device_used_peak_sampled_bytes') or 0)),
        'adapter_sha256': sha_file(adapter_file),
        'weight_upload': False,
        'base_weights_edited': False,
        'deployed': False,
        'quality_improvement_measured': False,
    }
    return out


def save_summary(path: Path, obj: dict) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ReportError('Refusing to overwrite a report')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(obj, file, indent=2, allow_nan=False)
        file.write('\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--trainer-report', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--report-out', type=Path, required=True)
    args = p.parse_args(argv)
    if args.trainer_report.is_symlink() or args.trainer_report.stat().st_size > 1024 * 1024:
        p.error('Invalid local training report')
    with args.trainer_report.open(encoding='utf-8') as f:
        report = json.load(f)
    save_summary(args.report_out, summarize_run(report, args.adapter))


if __name__ == '__main__':
    main()
