from pathlib import Path
import pytest
from neral_control import corpus
ROOT=Path(__file__).resolve().parents[2]

def test_exact_inventory_and_commitments():
    r=corpus.validate(ROOT)
    assert r['benchmark_tasks']==100 and r['categories']==corpus.COUNTS
    assert r['new_sft_examples']==60 and r['total_sft_examples']==86
    assert not r['leakage']['flagged_pairs']
    assert r['independent_review_completed'] is False

def test_duplicate_detector():
    rows=[('train_new.prompt','a','A bounded arithmetic expression has exactly one supported operator here.'),
          ('dev.prompt','b','a bounded arithmetic expression has exactly one supported operator here!'),
          ('train_new.full','c','Unrelated completion accompanying the first prompt.'),
          ('dev.full','d','Another completely different rubric; prompt matches are screened separately.')]
    assert corpus.leakage(rows)['flagged_pairs']

def test_modified_prompt_rejected():
    rows,manifest=corpus.read_benchmark(ROOT);rows[0]['prompt']+=' unreviewed'
    with pytest.raises(ValueError):corpus.validate_benchmark(rows,manifest)

def test_changed_category_rejected():
    rows,manifest=corpus.read_benchmark(ROOT);rows[0]['category']='tools_math'
    with pytest.raises(ValueError):corpus.validate_benchmark(rows,manifest)

def test_language_witnesses():
    import subprocess
    left=['a','b'];right=list(left)
    assert left==right and left is not right
    assert bool('false') is True
    def parse_flag(s):
        if s not in {'true','false'}:raise ValueError('unknown flag')
        return s=='true'
    assert parse_flag('false') is False
    with pytest.raises(ValueError):parse_flag('unknown')
    calls=[];value={'x':0}.get('x',calls.append('called'))
    assert value==0 and calls==['called']
    calls.clear();mapping={'x':0}
    value=mapping['x'] if 'x' in mapping else calls.append('called')
    assert value==0 and calls==[]
    subprocess.run(['node','-e', '''
const a=require('node:assert/strict');
a.deepEqual([{ok:true},{ok:false}].filter(r=>{r.ok}),[]);
a.deepEqual([{ok:true},{ok:false}].filter(r=>r.ok===true),[{ok:true}]);
const broken=['10','11','12'].map(parseInt);
a.equal(broken[0],10);a(Number.isNaN(broken[1]));a.equal(broken[2],1);
a.deepEqual(['10','11','12'].map(s=>parseInt(s,10)),[10,11,12]);
a.equal(encodeURIComponent('rock&roll'),'rock%26roll');
'''],check=True,timeout=10,capture_output=True)
