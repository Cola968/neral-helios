"""Deterministic sandbox programs from fully source-certified finite questions.

No model output or expected answer is used to author the program. Parsing
must cover the entire original question; the independent source verifier is
still mandatory after execution. Unknown grammar fails closed.
"""
from __future__ import annotations
import ast
import re
from neral_python_post_source import original_python_program, parse_post_source


def source_program(question: str) -> str | None:
    if type(question) is not str:
        return None
    # Source-supplied Python is a different trust boundary: it has already
    # passed the strict original-program AST/allowlist parser and runs only in
    # Docker, with the original script rerun for independent comparison.
    original = original_python_program(question)
    if original is not None:
        return original
    task = parse_post_source(question)
    if task is None:
        return None
    vals = [int(x) for x in re.findall(r'(?<!\w)\d+(?!\w)', question)]
    family = task.family
    if family == 'modular_exponent':
        a, b, m = vals[:3]
        body = f'a,b,m={a},{b},{m}\nanswer=1\nfor _ in range(b):\n    answer=(answer*a)%m\nassert answer==pow(a,b,m)'
    elif family == 'divisor_count':
        n = vals[0]
        body = f'n={n}\nanswer=sum(n%i==0 for i in range(1,n+1))\nassert answer>=1'
    elif family == 'congruence_square_sum':
        n, m, residue = vals[-3:]
        body = (f'n,m,r={n},{m},{residue}\nanswer=0\nfor k in range(1,n+1):\n'
                '    if k%m==r:\n        answer+=k*k\nassert answer>=0')
    elif family == 'forbidden_stair_paths':
        n, banned = vals[0], vals[-1]
        body = (f'n,banned={n},{banned}\nways=[0]*(n+1)\nways[0]=1\n'
                'for i in range(1,n+1):\n    if i!=banned:\n'
                '        ways[i]=sum(ways[i-j] for j in (1,2,3) if i>=j)\n'
                'answer=ways[n]\nassert answer>=0')
    elif family == 'ring_switches':
        n, k = vals[:2]
        body = (f'import itertools\nn,k={n},{k}\nanswer=0\n'
                'for bits in itertools.product((0,1), repeat=n):\n'
                '    if sum(bits)==k and not any(bits[i] and bits[(i+1)%n] for i in range(n)):\n'
                '        answer+=1\nassert answer>=0')
    elif family == 'forbidden_rooks':
        n = vals[0]
        lo, hi = question.find('['), question.rfind(']')
        if lo < 0 or hi <= lo:
            return None
        try:
            blocked = ast.literal_eval(question[lo:hi+1])
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return None
        body = (f'import itertools\nn={n}\nblocked={repr(blocked)}\n'
                'answer=sum(1 for p in itertools.permutations(range(n)) '
                'if all((r,c) not in blocked for r,c in enumerate(p)))\n'
                'assert answer>=0')
    elif family == 'electric_charge':
        a, b = vals[:2]
        body = f'current,seconds={a},{b}\nanswer=current*seconds\nassert answer>=0'
    elif family == 'airline_load':
        f, capacity, empty = vals[:3]
        body = (f'flights,capacity,empty={f},{capacity},{empty}\n'
                'answer=sum(capacity-empty for _ in range(flights))\n'
                'assert 0<=answer<=flights*capacity')
    elif family == 'spring_energy':
        k, x = vals[:2]
        body = f'k,x={k},{x}\nanswer=(k*x*x)//2\nassert 2*answer==k*x*x'
    elif family == 'warehouse_stock':
        initial, days, incoming, outgoing = vals[:4]
        body = (f'initial,days,incoming,outgoing={initial},{days},{incoming},{outgoing}\n'
                'answer=initial\nfor _ in range(days):\n'
                '    answer+=incoming-outgoing\nassert answer>=0')
    else:
        return None
    program = body + '\nprint(answer)\n'
    if len(program.encode('utf8')) > 4096:
        return None
    return program
