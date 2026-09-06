from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ByteTokenizer:
    """A deterministic 260-token byte tokenizer.

    IDs:
      0 <pad>
      1 <bos>
      2 <eos>
      3 <sep>
      4..259 raw byte values 0..255
    """

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    sep_id: int = 3
    byte_offset: int = 4
    vocab_size: int = 260

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = [b + self.byte_offset for b in text.encode("utf-8")]
        if bos:
            ids.insert(0, self.bos_id)
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = bytearray()
        for idx in ids:
            idx = int(idx)
            if self.byte_offset <= idx < self.vocab_size:
                out.append(idx - self.byte_offset)
            elif not skip_special:
                marker = {
                    self.pad_id: b"<pad>",
                    self.bos_id: b"<bos>",
                    self.eos_id: b"<eos>",
                    self.sep_id: b"<sep>",
                }.get(idx, b"<?>")
                out.extend(marker)
        return out.decode("utf-8", errors="replace")
