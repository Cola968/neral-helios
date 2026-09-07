from types import SimpleNamespace

from benchmark_v5_long_context import (
    BPB_LENGTHS,
    NEEDLES,
    SINGLE_DISTANCES,
    _filler,
    _fits,
    suite_sha256,
)


def test_suite_definition_is_stable_shape():
    digest = suite_sha256()
    assert len(digest) == 64
    assert len(set(NEEDLES)) == len(NEEDLES) == 4
    assert SINGLE_DISTANCES == tuple(sorted(SINGLE_DISTANCES))
    assert BPB_LENGTHS[-1] == 4095


def test_filler_is_exact_in_utf8_bytes():
    for lang in ("de", "en"):
        for n in (1, 127, 512, 3800):
            assert len(_filler(n, lang).encode("utf8")) == n


def test_fit_check_respects_4096_byte_token_window():
    model = SimpleNamespace(cfg=SimpleNamespace(max_seq_len=4096))
    short_prefix = "Passcode: ORBIT-7K4Q.\n" + _filler(3800, "en") + "\nPasscode:"
    assert _fits(model, short_prefix, " ORBIT-7K4Q.", " NOVA-3M8P.")
    too_long = short_prefix + ("x" * 400)
    assert not _fits(model, too_long, " ORBIT-7K4Q.", " NOVA-3M8P.")
