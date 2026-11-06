"""Offline read-only Qwen7B versus experimental rank-32 NERAL adapter smoke.

Only fixed prompts; never executes generated code on the host. Optional Docker
Linux sandbox executes tests with no network, all caps dropped and hard limits.
No training, downloading, deployment or weights modification.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

SYSTEM = ('Du schreibst Python 3. Gib ausschließlich ausführbaren Python-Code aus, '
          'ohne Markdown, Erklärungen, Ein- oder Ausgabe und ohne fremde Bibliotheken.')
TASKS = [
    ('even', 'Schreibe is_even(n), die für eine ganze Zahl True zurückgibt, wenn sie gerade ist.',
     'assert is_even(0) is True\nassert is_even(2) is True\nassert is_even(-2) is True\nassert is_even(3) is False\n'),
    ('deduplicate', 'Schreibe deduplicate(items): Entferne doppelte Ganzzahlen aus einer Liste und erhalte die Reihenfolge ihres ersten Auftretens.',
     'assert deduplicate([3, 1, 3, 2, 1]) == [3, 1, 2]\nassert deduplicate([]) == []\nassert deduplicate([1, 1]) == [1]\n'),
    ('word_counts', 'Schreibe word_counts(text): Zähle durch Leerraum getrennte Wörter ohne Unterscheidung zwischen Groß- und Kleinschreibung. Satzzeichen bleiben Teil des Wortes.',
     'assert word_counts("Hallo hallo Welt") == {"hallo": 2, "welt": 1}\nassert word_counts("") == {}\nassert word_counts("A, a A,") == {"a,": 2, "a": 1}\n'),
    ('average', 'Korrigiere def average(xs): return sum(xs) / (len(xs) - 1). Bei einer leeren Liste soll ValueError ausgelöst werden.',
     'assert average([2, 4]) == 3\nassert average([5]) == 5\ntry:\n    average([])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError("ValueError erwartet")\n'),
    ('parse_port', 'Schreibe parse_port(value): Wandle eine Zeichenkette in eine ganze Zahl von 1 bis 65535 um. Ungültige Werte lösen ValueError aus.',
     'assert parse_port("80") == 80\nassert parse_port("65535") == 65535\nfor s in ("0", "65536", "abc"):\n    try:\n        parse_port(s)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError("ValueError erwartet: " + s)\n'),
    ('merge_ranges', 'Schreibe merge_ranges(ranges): Führe überlappende oder sich berührende Intervalle zusammen. Eingabe und Ausgabe sind Listen von Tupeln ganzer Zahlen mit Anfang <= Ende. Sortiere die Ausgabe aufsteigend.',
     'assert merge_ranges([(1, 3), (3, 5), (8, 9)]) == [(1, 5), (8, 9)]\nassert merge_ranges([(8, 9), (2, 4), (1, 3)]) == [(1, 4), (8, 9)]\nassert merge_ranges([]) == []\n'),
]


def extract_code(text: str) -> str:
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.I | re.S)
    return (match.group(1) if match else text).strip() + '\n'


def docker_command(image: str, directory: Path, container_name: str, cpu: int = 1) -> list[str]:
    if not re.fullmatch(r'python:3\.11-alpine(?:@[a-z0-9:]+)?', image):
        raise ValueError('Only the locally available pinned Python test image is allowed')
    return ['docker', 'run', '--pull', 'never', '--name', container_name,
            '--network', 'none', '--read-only', '--memory', '256m', '--cpus', str(cpu),
            '--pids-limit', '64', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
            '--mount', f'type=bind,src={directory.resolve()},dst=/work,readonly',
            '--workdir', '/work', '--env', 'PYTHONDONTWRITEBYTECODE=1', image,
            'python', '-I', '-B', '-c',
            "import sys; sys.path.insert(0, '/work'); exec(open('/work/check.py').read())"]


def redact_runtime_result(raw: dict) -> dict:
    return {'passed': raw.get('passed'), 'returncode': raw.get('returncode')}


def _docker_present(image: str) -> bool:
    try:
        cmd = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, timeout=12)
        return cmd.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _sandbox_test(code: str, tests: str, image: str, timeout: int) -> dict:
    name = 'neral-r32-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='neral-r32-check-') as tmp:
        directory = Path(tmp)
        (directory / 'solution.py').write_text(code, encoding='utf-8')
        (directory / 'check.py').write_text('from solution import *\n' + tests, encoding='utf-8')
        # chmod is meaningful on Linux; on Windows Docker Desktop handles the bind mount.
        try:
            directory.chmod(0o755)
            for p in directory.iterdir():
                p.chmod(0o644)
        except OSError:
            pass
        try:
            p = subprocess.run(docker_command(image, directory, name), capture_output=True,
                               timeout=timeout, check=False)
            if p.returncode in (125, 126, 127):
                return {'passed': None, 'returncode': p.returncode,
                        'execution_status': 'docker_error'}
            return {'passed': p.returncode == 0, 'returncode': p.returncode,
                    'execution_status': 'executed'}
        except subprocess.TimeoutExpired:
            return {'passed': False, 'returncode': None, 'execution_status': 'timeout'}
        except OSError:
            return {'passed': None, 'returncode': None, 'execution_status': 'docker_error'}
        finally:
            try:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True,
                               timeout=15, check=False)
            except (OSError, subprocess.SubprocessError):
                pass


def summary_from_rows(runs: dict) -> dict:
    out = {'schema': 1, 'suite': 'six_task_development_smoke', 'runs': {},
           'paired': {'comparable': 0, 'adapter_only': 0, 'baseline_only': 0,
                      'both_pass': 0, 'neither_pass': 0},
           'quality_gain_proven': False, 'independent_holdout': False,
           'weights_modified': False, 'trained': False, 'deployed': False}
    for label in ('basis', 'neral'):
        rows = runs[label]
        checked = [r for r in rows if r.get('passed') is not None]
        out['runs'][label] = {
            'generated': len(rows), 'tested': len(checked),
            'passed': sum(r['passed'] is True for r in checked) if checked else None,
            'pass_rate': round(sum(r['passed'] is True for r in checked) / len(checked), 4)
            if checked else None,
            'syntax_valid': sum(r.get('syntax_valid') is True for r in rows),
        }
    baseline = {r['task']: r for r in runs['basis']}
    adapter = {r['task']: r for r in runs['neral']}
    for task in sorted(set(baseline) & set(adapter)):
        a, b = baseline[task].get('passed'), adapter[task].get('passed')
        if a is None or b is None:
            continue
        out['paired']['comparable'] += 1
        key = ('both_pass' if a and b else 'baseline_only' if a else
               'adapter_only' if b else 'neither_pass')
        out['paired'][key] += 1
    return out


def evaluation_status(runs: dict, *, docker_ready: bool) -> str:
    """Do not confuse a locally installed Docker image with executed tests."""
    if not docker_ready:
        return 'GENERATED_ONLY'
    statuses = [row.get('execution_status') for rows in runs.values() for row in rows]
    evaluated = sum(status in ('executed', 'timeout', 'invalid_syntax') for status in statuses)
    if not evaluated:
        return 'GENERATED_ONLY'
    return 'EVALUATED_WITH_DOCKER' if evaluated == len(statuses) else 'PARTIALLY_EVALUATED'


def _model(base: Path):
    import torch
    from transformers import BitsAndBytesConfig, Qwen2ForCausalLM
    return Qwen2ForCausalLM.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False, token=False,
        use_safetensors=True, attn_implementation='sdpa', device_map={'': 0},
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))


def _generate_tasks(label: str, model, tokenizer, output: Path, image: str,
                    docker_ready: bool, timeout: int, max_new_tokens: int) -> list[dict]:
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval()
    rows = []
    for task_id, prompt, tests in TASKS:
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
        encoded = tokenizer.apply_chat_template(messages, chat_template=CHAT_TEMPLATE,
                                                 tokenize=True, add_generation_prompt=True,
                                                 return_tensors='pt').to('cuda:0')
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(encoded, attention_mask=torch.ones_like(encoded),
                                       do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        answer = tokenizer.decode(generated[0, encoded.shape[-1]:], skip_special_tokens=True)
        code = extract_code(answer)
        (output / f'{label}_{task_id}.py').write_text(code, encoding='utf-8')
        row = {'task': task_id, 'seconds': round(seconds, 3),
               'answer_sha256': hashlib.sha256(code.encode()).hexdigest()}
        try:
            ast.parse(code)
            row['syntax_valid'] = True
        except (SyntaxError, ValueError):
            row['syntax_valid'] = False
        if not docker_ready:
            row.update(passed=None, execution_status='docker_unavailable')
        elif not row['syntax_valid']:
            row.update(passed=False, execution_status='invalid_syntax')
        else:
            row.update(_sandbox_test(code, tests, image, timeout))
        rows.append(row)
        print(f'{label}/{task_id}: generation completed; runtime={row["execution_status"]}', flush=True)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--adapter', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--summary-out', required=True, type=Path)
    p.add_argument('--image', default='python:3.11-alpine')
    p.add_argument('--timeout', type=int, default=30)
    p.add_argument('--max-new-tokens', type=int, default=256)
    args = p.parse_args(argv)
    if not 5 <= args.timeout <= 120 or not 16 <= args.max_new_tokens <= 512:
        p.error('Invalid bounded time/token count')
    if not re.fullmatch(r'python:3\.11-alpine(?:@[a-z0-9:]+)?', args.image):
        p.error('Unknown container image')
    if args.out.exists() or args.summary_out.exists() or args.out.is_symlink():
        p.error('Refusing to overwrite existing output')
    if args.base.is_symlink() or args.adapter.is_symlink() or not args.base.is_dir() or not args.adapter.is_dir():
        p.error('Trusted local base/adapter directories required')
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate_r32 import validate_adapter
    base = validate_hf_snapshot(args.base, verify_hashes=True)
    manifest = validate_adapter(args.adapter, base)
    if manifest['release_validated']:
        p.error('Expected experimental, not release-promoted, rank-32 pilot')
    import torch
    from transformers import Qwen2TokenizerFast
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('RTX CUDA BF16 required')
    free, _ = torch.cuda.mem_get_info(0)
    if free < 10_000 * 1024 * 1024:
        p.error('Less than 10 GiB GPU free; aborting without model load')
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(args.base), local_files_only=True,
                                                   trust_remote_code=False, token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    docker_ready = _docker_present(args.image)
    args.out.mkdir(parents=True)
    results = {'basis': [], 'neral': []}
    for label in ('basis', 'neral'):
        model = _model(args.base)
        try:
            if label == 'neral':
                model = PeftModel.from_pretrained(model, str(args.adapter),
                                                   is_trainable=False, local_files_only=True)
            results[label] = _generate_tasks(label, model, tokenizer, args.out, args.image,
                                              docker_ready, args.timeout, args.max_new_tokens)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    # All raw code remains local; only bounded scalar results are exported.
    detailed = {'schema': 1, 'base_revision': base['revision'],
                'adapter_sha256': manifest['files']['adapter_model.safetensors'],
                'docker_used': docker_ready, 'runs': results,
                'no_training': True, 'no_autopromotion': True}
    (args.out / 'ergebnisse.json').write_text(json.dumps(detailed, indent=2), encoding='utf-8')
    summary = summary_from_rows(results)
    summary.update(status=evaluation_status(results, docker_ready=docker_ready),
                   base_revision=base['revision'],
                   adapter_sha256=manifest['files']['adapter_model.safetensors'],
                   max_new_tokens=args.max_new_tokens)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open('x', encoding='utf-8') as dest:
        json.dump(summary, dest, indent=2, allow_nan=False)
    print('COMPARE_COMPLETE; CODE_STORED_LOCALLY_ONLY; QUALITY_GAIN_NOT_PROVEN', flush=True)


if __name__ == '__main__':
    main()
