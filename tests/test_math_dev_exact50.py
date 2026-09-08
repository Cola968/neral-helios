from math_dev_exact50 import normalize_output, select_balanced_subset
from raw_math50_data import CATEGORIES


def rows():
    out = []
    for category in CATEGORIES:
        for i in range(8):
            out.append({
                "category": category,
                "language": "de" if i % 2 == 0 else "en",
                "prompt": f"{category}-{i}?",
                "answer": str(i),
                "canonical": f"{category}:{i}",
            })
    return out


def test_subset_is_balanced_and_deterministic():
    a = select_balanced_subset(rows(), per_category=3, seed=123)
    b = select_balanced_subset(rows(), per_category=3, seed=123)
    assert [r["canonical"] for r in a] == [r["canonical"] for r in b]
    assert len(a) == 3 * len(CATEGORIES)
    for category in CATEGORIES:
        assert sum(r["category"] == category for r in a) == 3


def test_subset_rejects_insufficient_category():
    data = [r for r in rows() if not (r["category"] == CATEGORIES[0] and int(r["answer"]) >= 2)]
    try:
        select_balanced_subset(data, per_category=3)
    except ValueError as exc:
        assert "not enough rows" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_normalization_is_strict_except_surrounding_whitespace():
    assert normalize_output(" 42\n") == "42"
    assert normalize_output("answer: 42") != "42"
