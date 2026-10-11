from __future__ import annotations
import re,unicodedata

SECRET_OR_CONTACT=re.compile(
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:sk-|gh[pousr]_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{16,}|"
    r"AKIA[A-Z0-9]{16}|-----BEGIN .*PRIVATE KEY",
    re.I,
)
CONTROL_RE=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
CHAR_RUN_RE=re.compile(r"(.)\1{15,}",re.S)

def clean_human_text(value):
    if not isinstance(value,str):
        return None,"not_text"
    text=unicodedata.normalize("NFC",value).strip()
    if not text:
        return None,"empty"
    if "\ufffd" in text or CONTROL_RE.search(text):
        return None,"encoding_or_control"
    if SECRET_OR_CONTACT.search(text):
        return None,"contact_or_secret"
    if CHAR_RUN_RE.search(text):
        return None,"character_run"
    return text,None

def all_texts_safe(texts):
    for text in texts:
        _,reason=clean_human_text(text)
        if reason:
            return False,reason
    return True,None
