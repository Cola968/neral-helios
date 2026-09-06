import torch
from config import NeralConfig
from model import Neral20M
from tokenizer import ByteTokenizer

cfg = NeralConfig()
model = Neral20M(cfg)
report = model.parameter_report()
print(report)
assert report["total"] == 20_587_008, report

tok = ByteTokenizer()
text = "Neral ✓ Berlin"
ids = tok.encode(text, bos=True, eos=True)
assert tok.decode(ids) == text
x = torch.randint(0, cfg.vocab_size, (2, 32))
logits, loss = model(x, x)
assert logits.shape == (2, 32, cfg.vocab_size)
assert torch.isfinite(loss)
print("smoke test OK")
