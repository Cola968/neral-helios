"""NERAL's explicit local quantitative tool, not an LLM benchmark or fine-tune.

Bounded full-match inputs only. Fraction arithmetic avoids float errors.
No eval, exec, network, model loading or arbitrary natural-language guessing.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import re


_NUM=r"[+-]?\d{1,6}"
_FRAC=r"([+-]?\d{1,6})/(\d{1,5})"


def _decimal_fraction(value:Fraction)->str:
    """Exact finite decimal if possible, or a reduced rational if not."""
    n,d=value.numerator,value.denominator
    if d==1:return str(n)
    factors=d
    for p in (2,5):
        while factors%p==0:
            factors//=p
    if factors!=1:
        return str(value)
    scale=0
    temp=d
    while temp>1:
        if temp%2==0:temp//=2
        elif temp%5==0:temp//=5
        scale+=1
    power=10**scale
    scaled=n*(power//d)
    sign="-" if scaled<0 else ""
    digits=str(abs(scaled)).zfill(scale+1)
    whole=digits[:-scale] if scale else digits
    decimals=digits[-scale:].rstrip("0") if scale else ""
    return sign+whole+(","+decimals if decimals else "")


def _result(value:str,method:str)->dict:
    return {"text":value,"exact":True,"tool_used":"neral_quant_exact",
            "method":method}


def _money(s:str)->dict|None:
    m=re.fullmatch(r"Berechne Endpreis:\s*(\d{1,6})(?:[.,](\d{1,2}))?\s*EUR,\s*"
                   r"Rabatt (\d{1,3})%,\s*Aufschlag (\d{1,3})%",s,re.IGNORECASE)
    if not m:return None
    whole,cents,disc,tax=m.groups()
    euros=int(whole)
    if euros>999999:return None
    cent=int((cents or "0").ljust(2,"0"))
    disc,tax=int(disc),int(tax)
    if disc>100 or tax>200:return None
    original=euros*100+cent
    exact_cents=Fraction(original*(100-disc)*(100+tax),10_000)
    cents_rounded=(2*exact_cents.numerator+exact_cents.denominator)//(2*exact_cents.denominator)
    return _result(f"{cents_rounded//100},{cents_rounded%100:02d} EUR","decimal_half_up")


def _fuel(s:str)->dict|None:
    m=re.fullmatch(r"Berechne Treibstoff:\s*(\d{1,6}) kg/h,\s*Flug (\d{1,4}) min,\s*"
                   r"Reserve (\d{1,4}) min,\s*Rollen (\d{1,6}) kg",s,re.IGNORECASE)
    if not m:return None
    rate,flight,reserve,roll=map(int,m.groups())
    if not (0<rate<=100000 and 0<=flight<=1440
            and 0<=reserve<=300 and 0<=roll<=100000):
        return None
    total=Fraction(rate*(flight+reserve),60)+roll
    return _result(_decimal_fraction(total)+" kg","rational_fuel")


def _arrival(s:str)->dict|None:
    m=re.fullmatch(r"Berechne Ankunftszeit:\s*(\d{2}):(\d{2}) UTC([+-]\d{1,2}),\s*"
                   r"Flug (\d{1,4}) min,\s*Ziel UTC([+-]\d{1,2})",s,re.IGNORECASE)
    if not m:return None
    hour,minute,src,duration,dst=map(int,m.groups())
    if (hour>23 or minute>59 or not -12<=src<=14 or not -12<=dst<=14
            or not 0<=duration<=2880):
        return None
    day,clock=divmod(60*hour+minute+duration+(dst-src)*60,1440)
    return _result(f"{clock//60:02d}:{clock%60:02d} (Tag {day:+d})","timezone_minutes")


def _fractions(s:str)->dict|None:
    m=re.fullmatch(r"Berechne Bruch:\s*"+_FRAC+r"\s*([+\-*/:])\s*"+_FRAC,s,re.IGNORECASE)
    if not m:return None
    a,b,op,c,d=m.groups()
    a,b,c,d=map(int,(a,b,c,d))
    if (not 1<=b<=10000 or not 1<=d<=10000 or
            abs(a)>1000000 or abs(c)>1000000):
        return None
    x,y=Fraction(a,b),Fraction(c,d)
    if op=="+":value=x+y
    elif op=="-":value=x-y
    elif op=="*":value=x*y
    else:
        if y==0:return None
        value=x/y
    if max(value.numerator.bit_length(),value.denominator.bit_length())>128:
        return None
    return _result(str(value),"rational_arithmetic")


# Each factor converts an exact unit quantity to a common SI base unit.
# Case matters: m (metre), min (minute), Pa (pascal), kPa (kilopascal).
# No temperature offsets: incompatible/unsupported dimensions fail closed.
_UNITS = {
    "mm": ("length", Fraction(1, 1000)),
    "cm": ("length", Fraction(1, 100)),
    "m": ("length", Fraction(1)),
    "km": ("length", Fraction(1000)),
    "ms": ("time", Fraction(1, 1000)),
    "s": ("time", Fraction(1)),
    "min": ("time", Fraction(60)),
    "h": ("time", Fraction(3600)),
    "g": ("mass", Fraction(1, 1000)),
    "kg": ("mass", Fraction(1)),
    "t": ("mass", Fraction(1000)),
    "m/s": ("speed", Fraction(1)),
    "km/h": ("speed", Fraction(5, 18)),
    "kt": ("speed", Fraction(1852, 3600)),
    "mph": ("speed", Fraction(1609344, 3600000)),
    "Pa": ("pressure", Fraction(1)),
    "hPa": ("pressure", Fraction(100)),
    "kPa": ("pressure", Fraction(1000)),
    "bar": ("pressure", Fraction(100000)),
    "J": ("energy", Fraction(1)),
    "Wh": ("energy", Fraction(3600)),
    "kWh": ("energy", Fraction(3600000)),
    "W": ("power", Fraction(1)),
    "kW": ("power", Fraction(1000)),
    "N": ("force", Fraction(1)),
    "kN": ("force", Fraction(1000)),
}
_UNIT_REQUEST = re.compile(
    r"(?i:wandle|konvertiere|rechne|convert)\s+"
    r"(?P<amount>[+-]?\d{1,8}(?:[.,]\d{1,6})?)\s+"
    r"(?P<source>[A-Za-z/]{1,5})\s+(?i:in|to)\s+"
    r"(?P<target>[A-Za-z/]{1,5})(?:\s+(?i:um))?[?.]?\Z"
)


def _convert_units(s: str) -> dict | None:
    """Independent, bounded SI conversion, no inference or user code."""
    match = _UNIT_REQUEST.fullmatch(s)
    if match is None:
        return None
    source = _UNITS.get(match["source"])
    target = _UNITS.get(match["target"])
    if source is None or target is None or source[0] != target[0]:
        return None
    # Decimal -> rational retains exact user-entered decimal values.
    amount = Fraction(Decimal(match["amount"].replace(",", ".")))
    result = amount * source[1] / target[1]
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 128:
        return None
    return _result(_decimal_fraction(result) + " " + match["target"],
                   "dimensional_rational_conversion")


def calculate_explicit(prompt:str)->dict|None:
    if (not isinstance(prompt,str) or not 1<=len(prompt)<=200
            or "\n" in prompt or "\r" in prompt):
        return None
    s=prompt.strip()
    if not s:return None
    for calculator in (_money,_fuel,_arrival,_fractions,_convert_units):
        answer=calculator(s)
        if answer is not None:
            return answer
    return None
