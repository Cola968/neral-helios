#!/usr/bin/env python3
"""Read-only algebra diagnostic: same 8 frozen equations, 3 output strategies, 3 models.

This is an explanatory DEVELOPMENT probe, not a blind release holdout. It does
not train, execute generated code, promote adapters, or upload model answers.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_math_v2_eval import cases, ensure_pinned_model_runtime
from scripts.neral_math_eval import normalized_number, _load_model
from scripts.neral_v2_generalization_probe import (
    EXPECTED_BASE_REVISION, EXPECTED_OLD_ADAPTER_SHA256, EXPECTED_V2_ADAPTER_SHA256,
)

MODES = ('direct', 'silent_steps', 'two_step')
MODELS = ('base', 'old_rank32', 'new_rank16_v2')
IDS = tuple(f'novel-{i:03d}' for i in range(41, 49))
FROZEN_FAILURES = {
    'novel-041': '53', 'novel-043': '8', 'novel-044': '31',
    'novel-045': '19', 'novel-046': '41', 'novel-048': '42',
}
EQUATION = re.compile(r'(-?\d+)x \+ \((-?\d+)\) = (-?\d+)')
X_LINE = re.compile(r'x\s*=\s*([+-]?\d+(?:/\d+)?)\s*')
AX_LINE = re.compile(r'([+-]?\d+)x\s*=\s*([+-]?\d+)\s*')
SYSTEM = ('Du bist NERAL. Beachte genau die Antwortanweisung. '
          'Berechne Gleichungen ohne externe Werkzeuge; erfinde keine Tool-Aufrufe.')


def derive_equations():
    """Read original generator's development cases; verify their algebra independently."""
    selected = [item for item in cases() if item['id'] in IDS]
    if [x['id'] for x in selected] != list(IDS):
        raise ValueError('Frozen eight equations missing or reordered')
    result = []
    for item in selected:
        matches = EQUATION.findall(item['prompt'])
        if len(matches) != 1:
            raise ValueError('Unexpected equation prompt format')
        a, b, c = map(int, matches[0])
        if not a:
            raise ValueError('Zero equation coefficient')
        x = Fraction(c - b, a)
        if x.denominator != 1 or str(x.numerator) != item['expected']:
            raise ValueError('Frozen expected value differs from independently solved equation')
        if item['id'] in FROZEN_FAILURES and item['expected'] != FROZEN_FAILURES[item['id']]:
            raise ValueError('Known six failure cases changed')
        result.append({**item, 'a': a, 'b': b, 'c': c, 'expression': matches[0]})
    return result


def build_variants(item):
    original = item['prompt']
    expression = f"{item['a']}x + ({item['b']}) = {item['c']}"
    return {
        'direct': original,
        'silent_steps': ('Löse '+expression+'. Führe die Umformung und Division '
                         'gedanklich sorgfältig aus. Nur den Zahlenwert ausgeben.'),
        'two_step': ('Löse '+expression+'. Zeige genau einen Zwischenschritt in der '
                     'Form "ax = Zahl" und dann in einer neuen Zeile "x = Zahl". '
                     'Keine anderen Wörter.'),
    }


def _fraction_or_none(s):
    if not s:
        return None
    try:
        return Fraction(s.replace(',', '.'))
    except (ValueError, ZeroDivisionError):
        return None


def grade_reply(item, mode, raw):
    """Return (mathematical final value correct, requested format valid, algebra valid)."""
    if mode not in MODES or not isinstance(raw, str):
        raise ValueError('Unknown mode or invalid model output')
    text = raw.strip()
    direct = normalized_number(text)
    final = _fraction_or_none(direct)
    if final is None:
        # Diagnose also extracts a final value from malformed outputs, without
        # awarding format compliance to answers containing extra explanation.
        m = re.search(r'(?<!\w)x\s*=\s*([+-]?\d+(?:/\d+)?)\s*\Z', text)
        if m:
            final = _fraction_or_none(m.group(1))
    correct = final == Fraction(item['expected'])
    if mode != 'two_step':
        return bool(correct), direct is not None, False
    lines = text.splitlines()
    if len(lines) != 2:
        return bool(correct), False, False
    step = AX_LINE.fullmatch(lines[0].strip())
    answer = X_LINE.fullmatch(lines[1].strip())
    algebra = bool(step and int(step.group(1)) == item['a']
                   and int(step.group(2)) == item['c'] - item['b'])
    format_valid = bool(step and answer and algebra and
                        _fraction_or_none(answer.group(1)) ==
                        Fraction(item['expected']))
    return bool(correct), format_valid, algebra


def case_digest(rows):
    public = [{'id': r['id'], 'prompt': r['prompt'], 'expected': r['expected']}
              for r in rows]
    return hashlib.sha256(json.dumps(public, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def summarize(results, equations):
    if set(results) != set(MODELS) or [r['id'] for r in equations] != list(IDS):
        raise ValueError('Missing models or frozen equations')
    counts = {}
    for model in MODELS:
        if set(results[model]) != set(MODES):
            raise ValueError('Missing prompt strategy')
        counts[model] = {}
        for mode in MODES:
            rows = results[model][mode]
            if len(rows) != len(IDS) or {v.get('id') for v in rows} != set(IDS):
                raise ValueError('Incomplete case results')
            counts[model][mode] = {
                'correct': sum(x['correct'] is True for x in rows),
                'format_valid': sum(x['format_valid'] is True for x in rows),
                'algebra_valid': sum(x['algebra_valid'] is True for x in rows),
                'total': len(rows),
            }
    return {
        'schema': 1, 'status': 'NERAL_V3_EQUATION_REASONING_DEV_DIAGNOSTIC',
        'cases': len(equations), 'prompt_modes': list(MODES),
        'catalog_sha256': case_digest(equations), 'models': counts,
        'former_v2_failed_ids': sorted(FROZEN_FAILURES),
        'general_improvement_proven': False,
        'training_performed': False, 'weights_promoted': False,
        'independent_holdout': False,
        'note': 'Eight already inspected development cases; prompt-level diagnosis only.'
    }


def _infer(model, tokenizer, equations):
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval()
    output = {mode: [] for mode in MODES}
    for item in equations:
        for mode, prompt in build_variants(item).items():
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, chat_template=CHAT_TEMPLATE, tokenize=True,
                add_generation_prompt=True, return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                generated = model.generate(
                    input_ids, attention_mask=torch.ones_like(input_ids),
                    do_sample=False, max_new_tokens=70,
                    pad_token_id=tokenizer.pad_token_id)
            answer = tokenizer.decode(
                generated[0, input_ids.shape[-1]:], skip_special_tokens=True)
            correct, valid, algebra = grade_reply(item, mode, answer)
            output[mode].append({
                'id': item['id'], 'correct': correct, 'format_valid': valid,
                'algebra_valid': algebra,
                'answer_sha256': hashlib.sha256(answer.encode('utf-8')).hexdigest(),
                'local_answer': answer[:300],
            })
            print(f"{item['id']}/{mode}: correct={correct}, format={valid}, algebra={algebra}",
                  flush=True)
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('base', 'old-r32', 'new-r16', 'trainer-root', 'out', 'summary'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink() or a.summary.exists():
        p.error('Fresh output paths required')
    for path in (a.base, a.old_r32, a.new_r16):
        if not path.is_dir() or path.is_symlink():
            p.error('Local pinned model directory missing or symlinked')
    equations = derive_equations()
    ensure_pinned_model_runtime(a.trainer_root)
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate import validate_adapter as validate_r16
    from neral_adapter_candidate_r32 import validate_adapter as validate_r32
    snapshot = validate_hf_snapshot(a.base, verify_hashes=True)
    old = validate_r32(a.old_r32, snapshot)
    new = validate_r16(a.new_r16, snapshot)
    if (snapshot.get('revision') != EXPECTED_BASE_REVISION or
        old['files']['adapter_model.safetensors'] != EXPECTED_OLD_ADAPTER_SHA256 or
        new['files']['adapter_model.safetensors'] != EXPECTED_V2_ADAPTER_SHA256 or
        old['release_validated'] or new['release_validated']):
        p.error('Model hash does not match frozen experimental baseline')
    import torch
    from peft import PeftModel
    from transformers import Qwen2TokenizerFast
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('CUDA BF16 not available')
    free, _ = torch.cuda.mem_get_info(0)
    if free < 9000*1024**2:
        p.error('GPU memory below safe inference threshold')
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        str(a.base), local_files_only=True, trust_remote_code=False, token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = {}
    a.out.mkdir(exist_ok=False)
    for label, adapter in [('base', None), ('old_rank32', a.old_r32),
                           ('new_rank16_v2', a.new_r16)]:
        model = _load_model(a.base)
        try:
            if adapter is not None:
                model = PeftModel.from_pretrained(
                    model, str(adapter), is_trainable=False, local_files_only=True)
            results[label] = _infer(model, tokenizer, equations)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    # Exact raw model completions are private to the PC; never upload this file.
    (a.out/'local_outputs.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report = summarize(results, equations)
    report.update(base_revision=snapshot['revision'],
                  old_adapter_sha256=EXPECTED_OLD_ADAPTER_SHA256,
                  new_adapter_sha256=EXPECTED_V2_ADAPTER_SHA256)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
