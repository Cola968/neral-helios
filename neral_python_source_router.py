"""Trusted independent full-source Python check multiplexor.

Generated-code assertions are only syntactic execution checks. Every
promotion requires either a deterministic finite-domain source parse or a
strict exact-arithmetic source parse independent of model-authored code.
"""
from __future__ import annotations
from neral_python_logic_verifier import independent_python_certificate
from neral_python_math_verifier import independent_arithmetic_certificate

def verify_python_full_source(*, source:str, code:str, output:str,
                               canonical:dict|None)->dict:
    logic=independent_python_certificate(
        source=source,code=code,output=output,canonical=canonical)
    if type(logic) is dict and logic.get("verified") is True:
        return logic
    arithmetic=independent_arithmetic_certificate(
        source=source,code=code,output=output,canonical=canonical)
    if type(arithmetic) is dict and arithmetic.get("verified") is True:
        return arithmetic
    return {"verified":False,"reason":"independent_proof_missing"}
