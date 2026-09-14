from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontier350"))

from v7_1_policy import choose_io_profile, update_batch_chain
from v7_policy import semantic_resume_contract


def test_streaming_batch_chain_is_byte_identical_to_legacy_concat():
    previous='ab'*32
    domain='math_reasoning'
    count=6144
    rows=np.arange(4*513,dtype=np.int64).reshape(4,513)
    expected=hashlib.sha256(
        bytes.fromhex(previous)+domain.encode()+count.to_bytes(8,'little')+rows.tobytes()
    ).hexdigest()
    assert update_batch_chain(previous,domain,count,rows)==expected


def test_fast3_input_flags_are_execution_only_for_exact_resume():
    fast3={'lr':2.6e-5,'data_sha256':{'main':{'x':'abc'}},'compile_mode':'max-autotune-no-cudagraphs',
           'fast_replay_only':True,'fused_replay_main':True,'v71_coalesced_input':True,
           'v71_input_transfer':'direct-pageable'}
    fallback={**fast3,'compile_mode':'default','fused_replay_main':False,
              'v71_coalesced_input':False,'v71_input_transfer':'legacy-pin'}
    assert semantic_resume_contract(fast3)==semantic_resume_contract(fallback)


def test_io_gate_selects_fastest_exact_candidate():
    report={
        'legacy_split':{'median_us':100.0,'peak_reserved_gib':1.0},
        'coalesced_pin':{'median_us':92.0,'peak_reserved_gib':1.05,'exact':True},
        'coalesced_pageable':{'median_us':80.0,'peak_reserved_gib':1.10,'exact':True},
    }
    selected,detail=choose_io_profile(report,min_speedup=1.05,max_extra_reserved_gib=.25)
    assert selected=='coalesced-pageable'
    assert detail['selected_profile']==selected


def test_io_gate_rejects_nonexact_or_slow_candidates():
    report={
        'legacy_split':{'median_us':100.0,'peak_reserved_gib':1.0},
        'coalesced_pin':{'median_us':80.0,'peak_reserved_gib':1.0,'exact':False},
        'coalesced_pageable':{'median_us':98.0,'peak_reserved_gib':1.0,'exact':True},
    }
    selected,_=choose_io_profile(report,min_speedup=1.05,max_extra_reserved_gib=.25)
    assert selected=='off'


def test_io_gate_rejects_memory_regression():
    report={
        'legacy_split':{'median_us':100.0,'peak_reserved_gib':1.0},
        'coalesced_pin':{'median_us':80.0,'peak_reserved_gib':1.5,'exact':True},
        'coalesced_pageable':{'median_us':75.0,'peak_reserved_gib':1.3,'exact':True},
    }
    selected,_=choose_io_profile(report,min_speedup=1.05,max_extra_reserved_gib=.25)
    assert selected=='off'
