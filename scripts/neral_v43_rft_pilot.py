"""CPU-only stratification and prompt-level leakage checks for the 200-row pilot.

Selection consumes already verified RFT and independently reviewed replay rows;
this module does not certify model answers or review provenance on its own.
"""
from collections import Counter
import hashlib
import random

REPLAY = dict.fromkeys(('everyday', 'knowledge', 'math', 'code'), 15)
RFT = {'sympy': 50, 'z3': 45, 'code': 45}
MIN_REPAIRS = 20


def prompt_hash(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('missing prompt')
    return hashlib.sha256(' '.join(text.split()).encode('utf8')).hexdigest()


def validate_pool(items):
    seen = set()
    for item in items:
        m = item['meta']
        key = m['task_hash']
        if key in seen:
            raise ValueError('duplicate source task')
        seen.add(key)
        if m['kind'] == 'rft':
            if (m.get('proof') not in RFT or m.get('source') != 'fresh_catalog'
                    or type(m.get('repair')) is not bool
                    or m['category'] != ('code' if m['proof'] == 'code' else 'math')):
                raise ValueError('invalid RFT provenance')
        elif m['kind'] == 'replay':
            if (m['category'] not in REPLAY or m.get('reviewed') is not True
                    or m.get('source') != 'independent_replay'
                    or key != prompt_hash(item['instruction'])):
                raise ValueError('invalid replay provenance')
        else:
            raise ValueError('invalid pool type')


def audit_pilot(items):
    validate_pool(items)
    replay = Counter(x['meta']['category'] for x in items if x['meta']['kind'] == 'replay')
    rft = Counter(x['meta']['proof'] for x in items if x['meta']['kind'] == 'rft')
    repairs = sum(x['meta'].get('repair') is True for x in items if x['meta']['kind'] == 'rft')
    if len(items) != 200 or replay != REPLAY or rft != RFT or repairs < MIN_REPAIRS:
        raise ValueError('pilot category or repair quota not met')
    return {'replay_by_category': dict(replay), 'rft_by_proof': dict(rft), 'repair_rows': repairs}


def choose_pilot(rft, replay, seed):
    validate_pool(rft + replay)
    if any(x['meta']['kind'] != 'rft' for x in rft) or any(x['meta']['kind'] != 'replay' for x in replay):
        raise ValueError('pool identity mismatch')
    rng = random.Random(seed)
    def shuffled(items):
        items = sorted(items, key=lambda x: x['meta']['task_hash'])
        rng.shuffle(items)
        return items
    selected = []
    for category, count in REPLAY.items():
        pool = shuffled([x for x in replay if x['meta']['category'] == category])
        if len(pool) < count:
            raise ValueError('insufficient reviewed replay: ' + category)
        selected.extend(pool[:count])
    for proof, count in RFT.items():
        pool = shuffled([x for x in rft if x['meta']['proof'] == proof])
        if len(pool) < count:
            raise ValueError('insufficient verified RFT: ' + proof)
        # Repairs are actual accepted candidate rows, never fabricated negatives.
        pool.sort(key=lambda x: not x['meta']['repair'])
        selected.extend(pool[:count])
    audit_pilot(selected)
    rng.shuffle(selected)
    return selected


def holdout_hashes(items, protected):
    """Check a separately reviewed, private, fixed 60-row development holdout.

    Exact prompt overlap is only a lower bound on leakage; semantic and family
    independence still need owner review before the holdout is pinned.
    """
    seen = set()
    categories = Counter()
    for item in items:
        key = prompt_hash(item['prompt'])
        if key in seen or key in protected:
            raise ValueError('duplicate or protected holdout prompt')
        seen.add(key)
        categories[item['category']] += 1
    if categories != REPLAY:
        raise ValueError('development holdout must cover 15 prompts per category')
    return seen
