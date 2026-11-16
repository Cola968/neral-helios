from scripts.neral_v2_forgetting_dev24 import score
from scripts.neral_1_1_postverify_dev24_cases import build,manifest

def test_original_dev24_is_complete_and_never_training():
    assert manifest()["cases"]==24
    assert manifest()["no_training"] is True
    assert len({c["question_sha256"] for c in build()})==24

def test_exact_numeric_score_without_ambiguous_parsing():
    assert score("42",42)==(True,False)
    assert score("41",42)==(False,False)
    assert score('{"answer":42}',42)==(True,False)
    assert score('{"tool":"z3","spec":{},"answer":42}',42)==(True,False)
    assert score("the answer could be 42 or 43",42)==(False,True)
    assert score('{"answer":"42"}',42)==(False,True)
