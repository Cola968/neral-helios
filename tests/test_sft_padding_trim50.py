import torch

from tokenizer import ByteTokenizer
from train_instruct50 import encode_sft, make_sft_batch, trim_sft_padding


TOK = ByteTokenizer()


def row(i, prompt, answer):
    return {"id": str(i), "prompt": prompt, "answer": answer, "language": "de", "domain": "chat"}


def test_trim_removes_only_common_trailing_padding():
    rows = [row(1, "Hi", "Kurz."), row(2, "Etwas länger", "Eine etwas längere Antwort.")]
    seq = 256
    encoded = [encode_sft(r, seq) for r in rows]
    full_x = torch.stack([x for x, _ in encoded])
    full_y = torch.stack([y for _, y in encoded])
    x, y = trim_sft_padding(full_x, full_y)
    assert x.shape == y.shape
    assert x.shape[1] < seq
    assert torch.equal(x, full_x[:, : x.shape[1]])
    assert torch.equal(y, full_y[:, : y.shape[1]])
    assert not bool(((x[:, -1] == TOK.pad_id) & (y[:, -1] == -100)).all())


def test_trim_preserves_every_supervised_target_and_eos():
    rows = [row(1, "A", "B"), row(2, "Frage", "Antwort mit mehr Text")]
    seq = 128
    encoded = [encode_sft(r, seq) for r in rows]
    full_y = torch.stack([y for _, y in encoded])
    full_targets = [y[y != -100].tolist() for y in full_y]
    x, y = make_sft_batch(rows, seq, torch.device("cpu"))
    trimmed_targets = [z[z != -100].tolist() for z in y]
    assert trimmed_targets == full_targets
    assert all(targets[-1] == TOK.eos_id for targets in trimmed_targets)
    assert x.shape[1] < seq


def test_trim_is_noop_when_longest_row_uses_full_width():
    seq = 64
    # Find an answer length that exactly occupies the remaining causal input width.
    prompt = "x"
    base = row(1, prompt, "")
    prefix_len = len(TOK.encode("User: " + prompt + "\nAssistant:", bos=True))
    # answer encoding adds leading space, newline and EOS; ids[:-1] must reach seq.
    answer_chars = seq - prefix_len - 2
    assert answer_chars > 0
    r = row(1, prompt, "a" * answer_chars)
    full_x, full_y = encode_sft(r, seq)
    x, y = trim_sft_padding(full_x.unsqueeze(0), full_y.unsqueeze(0))
    assert x.shape[1] == seq
    assert torch.equal(x[0], full_x)
    assert torch.equal(y[0], full_y)


def test_invalid_empty_batch_is_rejected():
    x = torch.zeros((2, 32), dtype=torch.long)
    y = torch.full((2, 32), -100, dtype=torch.long)
    try:
        trim_sft_padding(x, y)
    except ValueError as exc:
        assert "no active tokens" in str(exc)
    else:
        raise AssertionError("empty SFT batch must fail")
