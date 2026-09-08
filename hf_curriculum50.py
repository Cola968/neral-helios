"""Deterministic CPU-only curriculum routing for approved Hugging Face Neral sources.

No external model is called. Source-specific quality checks are intentionally conservative:
FineWeb2 German is used for German/general coverage, FineWeb-Edu is routed toward educational
science/math/reasoning, and OASST2 accepts human DE/EN assistant turns only.
"""
from __future__ import annotations
import hashlib
from prepare_pretrain50_b2 import normalize_for_dedup
from hf_source_policy50 import assert_training_allowed
CLASSES=('science','math','code','reasoning','german')
TERMS={'science':('physics','chemistry','biology','astronomy','planet','orbit','energy','force','quantum','relativity','physik','chemie','biologie','astronomie','energie','kraft'),'math':('equation','theorem','algebra','geometry','probability','integral','derivative','matrix','gleichung','satz','algebra','geometrie','wahrscheinlichkeit','integral','ableitung','matrix'),'code':('python','javascript','typescript','function','class ','algorithm','compiler','programming','def ','return ','exception','programmier','algorithmus'),'reasoning':('therefore','because','implies','assume','derive','proof','step by step','daher','weil','folgt','angenommen','herleiten','beweis','schritt für schritt'),'german':(' der ',' die ',' das ',' und ',' ist ',' nicht ',' eine ',' mit ',' für ',' von ',' zu ')}
def lexical_scores(text):
 n=' '+normalize_for_dedup(text)+' ';words=max(1,len(n.split()));return {k:sum(n.count(t) for t in ts)*1000./words for k,ts in TERMS.items()}
def basic_quality(text,min_chars=300):
 if not isinstance(text,str) or len(text.strip())<min_chars:return False
 clean=text.strip();printable=sum(c.isprintable() or c in '\n\t' for c in clean)/max(1,len(clean));alpha=sum(c.isalpha() for c in clean)/max(1,len(clean));return printable>=.98 and alpha>=.35
def source_admission(*,source_id,text,language=None,edu_score=None,role=None,deleted=False,synthetic=False,human_rank=None):
 assert_training_allowed(source_id)
 if source_id=='HuggingFaceFW/fineweb-2':
  ok=language=='de' and basic_quality(text,400);reason='fineweb2_de_quality' if ok else 'reject_non_de_or_low_quality'
 elif source_id=='HuggingFaceFW/fineweb-edu':
  score=float(edu_score) if edu_score is not None else -1.;ok=basic_quality(text,400) and score>=3.;reason='fineweb_edu_score3plus' if ok else 'reject_low_edu_quality'
 elif source_id=='OpenAssistant/oasst2':
  ok=language in {'de','en'} and role=='assistant' and not deleted and not synthetic and basic_quality(text,80);reason='oasst2_human_assistant' if ok else 'reject_oasst2_policy'
 else:
  ok=basic_quality(text,300);reason='approved_generic_quality' if ok else 'reject_low_quality'
 return {'accepted':ok,'reason':reason,'human_rank':human_rank}
def route(text,source_id,language=None,edu_score=None,human_rank=None):
 s=lexical_scores(text)
 if language=='de':s['german']+=4.
 if source_id=='HuggingFaceFW/fineweb-edu' and edu_score is not None:
  b=max(0.,min(5.,float(edu_score)));s['science']+=.3*b;s['math']+=.3*b;s['reasoning']+=.2*b
 if source_id=='OpenAssistant/oasst2' and human_rank is not None:s['reasoning']+=max(0.,min(3.,float(human_rank)))
 primary=sorted(CLASSES,key=lambda k:(-s[k],k))[0];return {'primary':primary,'scores':s,'normalized_sha256':hashlib.sha256(normalize_for_dedup(text).encode()).hexdigest()}
