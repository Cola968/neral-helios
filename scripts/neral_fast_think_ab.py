"""Read-only LOCAL FAST-vs-THINK A/B for one preinstalled Ollama model.

This is a small public development diagnostic for the model's reasoning switch,
NOT the NERAL THINK verifier/tool pipeline, a blinded holdout, or proof of
better NERAL weights. No model download, training, promotion or remote API.
Raw prompts/answers stay in a create-only private folder outside the repo.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts.neral_v7_foundation_tournament import (
    DOMAINS, FAMILIES, MODEL_DIGEST, MODEL_TAG, OllamaLocal, Refused,
    _create_private_folder, _json, evaluate_response, generate_cases,
    task_digest, SYSTEM, SEED,
)

CASE_SEED = 20261109
ALLOWED_COUNTS = (6, 12, 24)
ALLOWED_BUDGETS = (768, 1024, 1536, 2048)

# Qwen's model card explicitly warns against greedy decoding in thinking mode.
# Distinct recommended policies measure real deployment configurations; this
# does NOT isolate the causal effect of the boolean think switch.
QWEN3_SAMPLING = {
    'FAST': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20},
    'THINK': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20},
}


class RecommendedOllamaLocal(OllamaLocal):
    """Only a preinstalled Qwen3-8B, no pull and no remote endpoint."""

    def chat(self, model, prompt, *, think, max_tokens):
        if model != 'qwen3:8b':
            raise Refused('Recommended settings pinned to qwen3:8b only')
        policy = QWEN3_SAMPLING['THINK' if think else 'FAST']
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': prompt},
            ],
            'think': think,
            'stream': False,
            'keep_alive': '2m',
            'options': {
                **policy, 'num_predict': max_tokens, 'num_ctx': 4096, 'seed': SEED,
            },
        }
        response = self._request('/api/chat', payload)
        msg = response.get('message')
        content = msg.get('content') if isinstance(msg, dict) else None
        done_reason = response.get('done_reason')
        n_tokens = response.get('eval_count', 0)
        elapsed = response.get('total_duration', 0)
        if (not isinstance(content, str) or len(content) > 16384
                or done_reason not in ('stop', 'length')
                or type(n_tokens) is not int or not 0 <= n_tokens < 100000
                or type(elapsed) is not int or not 0 <= elapsed < 10**15):
            raise Refused('Invalid Qwen3 local response')
        return {
            'content': content, 'reason': done_reason,
            'eval_count': n_tokens, 'duration_ns': elapsed,
        }


def pick_cases(count: int) -> list[dict]:
    """A cross-domain diagnostic; only 12 public task families, not unseen methods."""
    if type(count) is not int or count not in ALLOWED_COUNTS:
        raise Refused('Only 6, 12 or 24 balanced public development cases allowed')
    generated = generate_cases(seed=CASE_SEED)
    grouped = {(row['domain'], row['family'], row['id'].split('-')[-1]): row
               for row in generated}
    selected = []
    # Round robin by domain, then method; each short sample includes all domains.
    for suffix in ('02', '03'):
        for index in range(4):
            for domain in DOMAINS:
                selected.append(grouped[(domain, FAMILIES[domain][index], suffix)])
    return selected[:count]


def compare_modes(client, model: str, output: Path, *, cases_count: int = 6,
                  max_tokens: int = 1024) -> dict:
    if (not isinstance(model, str) or MODEL_TAG.fullmatch(model) is None
            or model != 'qwen3:8b'):
        raise Refused('Only an explicitly named local Qwen3 model is supported')
    if type(max_tokens) is not int or max_tokens not in ALLOWED_BUDGETS:
        raise Refused('Choose a fixed 768/1024/1536/2048 token budget')
    cases = pick_cases(cases_count)
    installed = client.inventory()
    digest = installed.get(model)
    if (not isinstance(digest, str) or MODEL_DIGEST.fullmatch(digest) is None):
        raise Refused('Requested model is not installed locally')
    folder = _create_private_folder(output)
    counts = {mode: Counter() for mode in ('FAST', 'THINK')}
    duration = Counter()
    tokens = Counter()
    pairs = Counter()
    with (folder / 'private-responses.jsonl').open('x', encoding='utf8') as out:
        for index, case in enumerate(cases):
            grades = {}
            # Balanced AB/BA order reduces warm-up and order effects.
            order = ('FAST', 'THINK') if index % 2 == 0 else ('THINK', 'FAST')
            for mode in order:
                raw = client.chat(model, case['prompt'], think=(mode == 'THINK'),
                                  max_tokens=max_tokens)
                if (not isinstance(raw, dict) or
                        not isinstance(raw.get('content'), str) or
                        raw.get('reason') not in ('stop', 'length') or
                        type(raw.get('eval_count')) is not int or
                        type(raw.get('duration_ns')) is not int or
                        raw['eval_count'] < 0 or raw['duration_ns'] < 0):
                    raise Refused('Invalid local inference telemetry')
                grade = evaluate_response(case, raw['content'],
                                          complete=raw['reason'] == 'stop')
                counts[mode][grade['status']] += 1
                tokens[mode] += raw['eval_count']
                duration[mode] += raw['duration_ns']
                grades[mode] = grade['status'] == 'correct'
                out.write(_json({
                    'id': case['id'], 'domain': case['domain'],
                    'family': case['family'], 'mode': mode,
                    'prompt_private': case['prompt'],
                    'answer_private': case['answer'],
                    'raw_private': raw['content'], 'grade': grade,
                    'done_reason': raw['reason']}) + '\n')
                out.flush()
            if grades['FAST'] and grades['THINK']:
                pairs['both_correct'] += 1
            elif grades['FAST']:
                pairs['fast_only'] += 1
            elif grades['THINK']:
                pairs['think_only'] += 1
            else:
                pairs['both_wrong_or_ungraded'] += 1

    if client.inventory().get(model) != digest:
        raise Refused('Local model digest changed; refusing successful summary')
    result = {
        'schema': 1, 'status': 'NERAL_FAST_THINK_DEVELOPMENT_DIAGNOSTIC',
        'model': model, 'model_digest': digest, 'case_seed': CASE_SEED,
        'suite_sha256': task_digest(cases), 'n_per_mode': len(cases),
        'token_budget_each': max_tokens,
        'sampling_policy': QWEN3_SAMPLING,
        'temperature': 'mode_specific_non_greedy',
        'num_ctx': 4096, 'same_system_prompt': True,
        'pairs': dict(pairs),
        'accuracy': {m: {'correct': counts[m]['correct'],
                         'wrong_final': counts[m]['wrong_final'],
                         'ungraded': counts[m]['ungraded'],
                         'incomplete': counts[m]['incomplete'],
                         'seconds': round(duration[m] / 1e9, 3),
                         'generated_tokens': tokens[m]}
                     for m in ('FAST', 'THINK')},
        'think_minus_fast_correct_cases': pairs['think_only'] - pairs['fast_only'],
        'tools_compared': False, 'model_weights_changed': False,
        'models_downloaded': False, 'model_superiority_proven': False,
        'release_validated': False, 'blinded_holdout': False,
        'limitations': ('This comparison changes both sampling settings and the thinking switch; ' 
                        'therefore it measures recommended configurations, not the isolated flag. ' 
                        'Public, template-generated small development test of the '
                        'Ollama think switch, not the NERAL verifier and not an '
                        'independent generalization or release benchmark.'),
    }
    (folder / 'scalar-summary.json').write_text(_json(result) + '\n', encoding='utf8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', required=True,
                        help='Exact already installed Ollama name, e.g. qwen3:8b')
    parser.add_argument('--out', type=Path, required=True,
                        help='New private folder OUTSIDE the repository')
    parser.add_argument('--cases', type=int, choices=ALLOWED_COUNTS, default=6)
    parser.add_argument('--tokens', type=int, choices=ALLOWED_BUDGETS, default=1024)
    args = parser.parse_args(argv)
    report = compare_modes(RecommendedOllamaLocal(), args.model, args.out,
                           cases_count=args.cases, max_tokens=args.tokens)
    print(_json({'status': report['status'], 'model': report['model'],
                 'n_per_mode': report['n_per_mode'],
                 'accuracy': report['accuracy'], 'pairs': report['pairs'],
                 'model_superiority_proven': False}))


if __name__ == '__main__':
    main()
