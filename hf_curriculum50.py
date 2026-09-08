"""Deterministic CPU-only curriculum routing for approved Hugging Face Neral sources."""
from __future__ import annotations
import hashlib
from prepare_pretrain50_b2 import normalize_for_dedup
CLASSES=('science','math','code','reasoning','german')
TERMS={'science':('physics','chemistry','biology','astronomy','planet','orbit','energy','force','quantum','relativity','physik','chemie','biologie','astronomie','energie','kraft'),'math':('equation','theorem','algebra','geometry','probability','integral','derivative','matrix','gleichung','satz','algebra','geometrie','wahrscheinlichkeit','integral','ableitung','matrix'),'code':('python','javascript','typescript','function','class ','algorithm','compiler','programming','def ','return ','exception','programmier','algorithmus'),'reasoning':('therefore','because','implies','assume','derive','proof','step by step','daher','weil','folgt','angenommen','herleiten','beweis','schritt für schritt'),'german':(' der ',' die ',' das ',' und ',' ist ',' nicht ',' eine ',' mit ',' für ',' von ',' zu ')}
def lexical_scores(text):
 n=' '+normalize_for_dedup(text)+' ';words=max(1,len(n.split()));return {k:sum(n.count(t) for t in ts)*1000./words for k,ts in TERMS.items()}
def route(text,source_id,language=None,edu_score=None,human_rank=None):
 s=lexical_scores(text)
 if language=='de':s['german']+=4.
 if source_id=='HuggingFaceFW/fineweb-edu' and edu_score is not None:
  b=max(0.,min(5.,float(edu_score)));s['science']+=.3*b;s['math']+=.3*b;s['reasoning']+=.2*b
 if source_id=='OpenAssistant/oasst2' and human_rank is not None:s['reasoning']+=max(0.,min(3.,float(human_rank)))
 primary=sorted(CLASSES,key=lambda k:(-s[k],k))[0];return {'primary':primary,'scores':s,'normalized_sha256':hashlib.sha256(normalize_for_dedup(text).encode()).hexdigest()}
