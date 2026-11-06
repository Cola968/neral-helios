"""Deterministic public 12-task development benchmark (NOT training examples)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

TASKS=[
("coding-01","Off-by-one accumulator","sum_upto",
"Return the sum of all integers from 1 through n inclusive; for n <= 0 return 0.",
"def sum_upto(n):\n    return sum(range(max(n, 0)))\n",
["assert sum_upto(0)==0","assert sum_upto(1)==1","assert sum_upto(3)==6","assert sum_upto(10)==55"]),
("coding-02","Optional input handling","first_or_none",
"Return the first item of a sequence; return None for None or empty input.",
"def first_or_none(items):\n    return items[0]\n",
["assert first_or_none(None) is None","assert first_or_none([]) is None",
 "assert first_or_none([4,5])==4","assert first_or_none(['a'])=='a'"]),
("coding-03","Dictionary grouping and accumulation","total_by_category",
"Group a list of {category,value} records by category and add the integer values; empty input gives {}.",
"def total_by_category(rows):\n    out = {}\n    for row in rows:\n        out[row['category']] = row['value']\n    return out\n",
["assert total_by_category([])=={}",
 "assert total_by_category([{'category':'a','value':2},{'category':'a','value':3}])=={'a':5}",
 "assert total_by_category([{'category':'a','value':-2},{'category':'b','value':5}])=={'a':-2,'b':5}",
 "assert total_by_category([{'category':'x','value':0},{'category':'x','value':4}])=={'x':4}"]),
("coding-04","Resilient JSON parsing","parse_amount",
"Parse a JSON text object and return its integer 'amount'; return None for invalid JSON, wrong types or missing amount.",
"import json\ndef parse_amount(raw):\n    data = json.loads(raw)\n    return data['amount']\n",
["assert parse_amount(" + repr('{"amount":7}') + ")==7", "assert parse_amount('{}') is None",
 "assert parse_amount('not-json') is None","assert parse_amount('[1,2]') is None"]),
("coding-05","Stable score sorting with ties","rank_items",
"Return item names sorted by descending integer score and, on ties, alphabetical ascending name.",
"def rank_items(items):\n    return [v['name'] for v in sorted(items, key=lambda x:x['name'])]\n",
["assert rank_items([])==[]",
 "assert rank_items([{'name':'B','score':2},{'name':'A','score':5}])==['A','B']",
 "assert rank_items([{'name':'Bob','score':3},{'name':'Amy','score':3}])==['Amy','Bob']",
 "assert rank_items([{'name':'Z','score':-1},{'name':'C','score':4},{'name':'A','score':4}])==['A','C','Z']"]),
("coding-06","Order-preserving deduplication","unique_stable",
"Return unique hashable values in their first-appearance order, not sorted or unordered.",
"def unique_stable(items):\n    return list(set(items))\n",
["assert unique_stable([])==[]","assert unique_stable([2,1,2,3,1])==[2,1,3]",
 "assert unique_stable(['b','a','b'])==['b','a']","assert unique_stable([0,0,0])==[0]"]),
("coding-07","Inclusive boundary filter","minimum_filter",
"Keep values greater than or equal to the threshold, preserving input order.",
"def minimum_filter(values, threshold):\n    return [v for v in values if v > threshold]\n",
["assert minimum_filter([],2)==[]","assert minimum_filter([1,2,3],2)==[2,3]",
 "assert minimum_filter([-2,-1,0],-1)==[-1,0]","assert minimum_filter([5,5],5)==[5,5]"]),
("coding-08","Chunking with remainder","chunked",
"Split list into consecutive chunks of positive size; include trailing short chunk. Return [] if size <= 0.",
"def chunked(items, size):\n    return [items[i:i+size] for i in range(0,len(items)//size*size,size)]\n",
["assert chunked([1,2,3,4,5],2)==[[1,2],[3,4],[5]]",
 "assert chunked([1],2)==[[1]]","assert chunked([],3)==[]","assert chunked([1,2,3],1)==[[1],[2],[3]]"]),
("coding-09","Whitespace-normalized slug","slugify",
"Strip leading/trailing whitespace, lowercase ASCII text and join internal whitespace-separated words with single '-'.",
"def slugify(text):\n    return text.lower().replace(' ','-')\n",
["assert slugify('  Hello  World ')== 'hello-world'","assert slugify('A\tB')=='a-b'",
 "assert slugify(' A ')=='a'","assert slugify('')==''"]),
("coding-10","Rolling averages","window_mean",
"Return the mean for each complete consecutive window of positive length; return [] for missing full windows.",
"def window_mean(values,size):\n    return [sum(values[i:i+size])/(size+1) for i in range(len(values)-size+1)]\n",
["assert window_mean([2,4,6],2)==[3,5]",
 "assert window_mean([5],1)==[5]","assert window_mean([1,2],3)==[]",
 "assert window_mean([0,0,0],2)==[0,0]"]),
("coding-11","Parentheses state machine","balanced_parens",
"Return True when parentheses are properly nested; ignore non-parenthesis characters. Premature closing fails.",
"def balanced_parens(text):\n    return text.count('(') == text.count(')')\n",
["assert balanced_parens('()')==True","assert balanced_parens(')(')==False",
 "assert balanced_parens('(a(b)c)')==True","assert balanced_parens('(()')==False"]),
("coding-12","Counter merge without overwrite","merge_counters",
"Return a new dictionary with counts from both dictionaries added for shared keys; do not mutate either input.",
"def merge_counters(left,right):\n    out=left.copy()\n    out.update(right)\n    return out\n",
["assert merge_counters({}, {})=={}",
 "assert merge_counters({'a':2},{'a':3})=={'a':5}",
 "assert merge_counters({'a':2},{'b':3})=={'a':2,'b':3}",
 "assert merge_counters({'x':-1},{'x':3,'y':0})=={'x':2,'y':0}"])
]


def caseset():
    return [{"id":cid,"title":title,"function":fn,"goal":goal,
             "buggy_source":source,"test_cases":tests}
            for cid,title,fn,goal,source,tests in TASKS]


def canonical(data):
    return json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")


def main():
    rows=caseset()
    out=Path("benchmarks/neral_agent_coding_suite.json")
    if out.exists():
        existing=json.loads(out.read_text(encoding="utf-8"))
        if existing!=rows:raise ValueError("Frozen benchmark suite differs; reviewed change required")
    else:out.write_bytes(canonical(rows)+b"\n")
    print(json.dumps({"cases":len(rows),"sha256":hashlib.sha256(canonical(rows)).hexdigest(),
                      "status":"PUBLIC_DEV_NOT_HOLDOUT"},sort_keys=True))

if __name__=="__main__": main()
