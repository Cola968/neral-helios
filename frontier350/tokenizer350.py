from __future__ import annotations

"""Stable byte-token contract for Neral 350M Frontier.

Token IDs are intentionally architecture-owned and independent of any foreign tokenizer.
0..255 are raw UTF-8 bytes; the remaining four IDs are reserved control tokens.
"""

from dataclasses import dataclass

BYTE_VOCAB = 256
BOS_ID = 256
EOS_ID = 257
SEP_ID = 258
PAD_ID = 259
VOCAB_SIZE = 260


@dataclass(frozen=True)
class FrontierByteTokenizer:
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    sep_id: int = SEP_ID
    pad_id: int = PAD_ID
    vocab_size: int = VOCAB_SIZE

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids: list[int] = []
        if bos:
            ids.append(self.bos_id)
        ids.extend(text.encode("utf-8", errors="strict"))
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, *, skip_special: bool = True) -> str:
        raw = bytearray()
        for value in ids:
            token = int(value)
            if 0 <= token < BYTE_VOCAB:
                raw.append(token)
            elif token in (self.bos_id, self.eos_id, self.sep_id, self.pad_id):
                if not skip_special:
                    raise ValueError("special tokens cannot be represented as raw UTF-8 bytes")
            else:
                raise ValueError(f"token outside Frontier vocabulary: {token}")
        return raw.decode("utf-8", errors="replace")

    def encode_document(self, text: str) -> list[int]:
        return self.encode(text, bos=True, eos=True)


TOKENIZER = FrontierByteTokenizer()


def contract() -> dict:
    return {
        "type": "utf8-byte-v1",
        "byte_ids": [0, 255],
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "sep_id": SEP_ID,
        "pad_id": PAD_ID,
        "vocab_size": VOCAB_SIZE,
    }
