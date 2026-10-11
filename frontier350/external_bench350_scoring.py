from __future__ import annotations
import decimal,re
NUM=re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
def numeric(x):
 vals=NUM.findall(str(x))
 if not vals:return None
 try:return decimal.Decimal(vals[-1].replace(",",""))
 except decimal.InvalidOperation:return None
def gsm_target(answer):return numeric(str(answer).split("####")[-1])
def gsm_response(text):return numeric(str(text).split("####")[-1] if "####" in str(text) else str(text))
def code_only(text):
 s=str(text).strip();m=re.search(r"\x60\x60\x60(?:python)?\s*(.*?)\x60\x60\x60",s,re.I|re.S);return (m.group(1) if m else s).strip()
