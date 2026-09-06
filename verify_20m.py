from config import NeralConfig
from model import Neral20M
m=Neral20M(NeralConfig())
n=sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Trainierbare Parameter: {n:,}")
assert n==20_587_008
print("OK: exakt Neral 20M Architektur")
