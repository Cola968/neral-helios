"""Bounded arithmetic tool, separate from model generation. No eval/exec."""
import ast,re
from fractions import Fraction

def arithmetic(expression:str)->dict:
    expression=re.sub(r'^\s*SOLVE\s*:\s*','',expression,flags=re.I).strip()
    if not expression or len(expression)>180:raise ValueError('Gib einen Rechenausdruck mit höchstens 180 Zeichen ein.')
    if not re.fullmatch(r'[0-9\s.+*/()%\-^]+',expression):raise ValueError('SOLVE unterstützt nur Zahlen, Klammern und + - * / // % **.')
    expression=expression.replace('^','**')
    try:tree=ast.parse(expression,mode='eval')
    except (SyntaxError,RecursionError) as e:raise ValueError('Ungültiger Rechenausdruck.') from e
    if len(list(ast.walk(tree)))>64:raise ValueError('Der Rechenausdruck ist zu komplex.')
    steps=[]
    def fmt(n):return str(n.numerator) if n.denominator==1 else f'{n.numerator}/{n.denominator}'
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            raw=ast.get_source_segment(expression,node)
            try:n=Fraction(raw)
            except (ValueError,ZeroDivisionError):raise ValueError('Ungültige Zahl.')
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            n=visit(node.operand);n=-n if isinstance(node.op,ast.USub) else n
        elif isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):n=a+b;op='+'
            elif isinstance(node.op,ast.Sub):n=a-b;op='-'
            elif isinstance(node.op,ast.Mult):n=a*b;op='*'
            elif isinstance(node.op,(ast.Div,ast.FloorDiv,ast.Mod)):
                if b==0:raise ValueError('Division durch null ist nicht definiert.')
                if isinstance(node.op,ast.Div):n=a/b;op='/'
                elif isinstance(node.op,ast.FloorDiv):n=Fraction(a//b);op='//'
                else:n=a%b;op='%'
            elif isinstance(node.op,ast.Pow):
                if b.denominator!=1 or abs(b)>12:raise ValueError('Nur ganzzahlige Exponenten von -12 bis 12 sind erlaubt.')
                if a==0 and b<=0:raise ValueError('Dieser Potenzausdruck wird nicht unterstützt.')
                n=a**int(b);op='**'
            else:raise ValueError('Operator nicht unterstützt.')
            steps.append(f'{fmt(a)} {op} {fmt(b)} = {fmt(n)}')
        else:raise ValueError('Ausdruck nicht unterstützt.')
        if abs(n.numerator)>10**24 or n.denominator>10**24:raise ValueError('Zahlenbereich überschritten.')
        return n
    answer=visit(tree.body)
    return {'expression':expression,'result':fmt(answer),'steps':steps,'engine':'bounded-fraction-arithmetic','model_generated':False}

def solve(expression:str)->dict:
    if not isinstance(expression,str):raise ValueError('Ein Textausdruck wird benötigt.')
    if '=' not in expression:return arithmetic(expression)
    expression=re.sub(r'^\s*SOLVE\s*:\s*','',expression,flags=re.I).strip()
    if len(expression)>180 or expression.count('=')!=1 or not re.fullmatch(r'[0-9x\s.+*/()=\-]+',expression):raise ValueError('Lineare Gleichung, z. B. 2*x + 3 = 11, eingeben.')
    # Reject ambiguous implicit multiplication even if parsing behavior changes.
    if re.search(r'(?:\d|\))\s*x\b', expression):
        raise ValueError('Multiplikation ausdrücklich als 2*x schreiben.')
    steps=[]
    def visit(n,source):
        if isinstance(n,ast.Name) and n.id=='x':return Fraction(1),Fraction(0)
        if isinstance(n,ast.Constant) and type(n.value) in [int,float]:
            try:value=Fraction(ast.get_source_segment(source,n))
            except (ValueError,ZeroDivisionError):raise ValueError('Ungültige Zahl.')
            if abs(value.numerator)>10**24 or value.denominator>10**24:raise ValueError('Zahlenbereich überschritten.')
            return Fraction(0),value
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            a,b=visit(n.operand,source);return (-a,-b) if isinstance(n.op,ast.USub) else (a,b)
        if not isinstance(n,ast.BinOp):raise ValueError('Nur lineare Gleichungen mit x werden unterstützt.')
        a,b=visit(n.left,source);c,d=visit(n.right,source)
        if isinstance(n.op,ast.Add):result=(a+c,b+d)
        elif isinstance(n.op,ast.Sub):result=(a-c,b-d)
        elif isinstance(n.op,ast.Mult):
            if a and c:raise ValueError('Nichtlinear: x*x wird noch nicht unterstützt.')
            result=(a*d+c*b,b*d)
        elif isinstance(n.op,ast.Div):
            if c or not d:raise ValueError('Division nur durch eine Konstante ungleich null.')
            result=(a/d,b/d)
        else:raise ValueError('Operator in linearer Gleichung nicht unterstützt.')
        if any(abs(v.numerator)>10**24 or v.denominator>10**24 for v in result):raise ValueError('Zahlenbereich überschritten.')
        return result
    sides=[]
    for side in expression.split('='):
        side=side.strip()
        try:tree=ast.parse(side,mode='eval')
        except (SyntaxError,RecursionError):raise ValueError('Ungültige Gleichung. Multiplikation als 2*x schreiben.')
        if len(list(ast.walk(tree)))>64:raise ValueError('Gleichung zu komplex.')
        sides.append(visit(tree.body,side))
    (a,b),(c,d)=sides;coefficient=a-c;constant=d-b
    steps=[f'Links: ({a})*x + ({b}); rechts: ({c})*x + ({d})',f'Zusammenfassen: ({coefficient})*x = {constant}']
    if coefficient==0:answer='Unendlich viele Lösungen' if constant==0 else 'Keine Lösung'
    else:
        x=constant/coefficient
        if abs(x.numerator)>10**24 or x.denominator>10**24:raise ValueError('Zahlenbereich überschritten.')
        answer=f'x = {x}';steps.append(f'Durch {coefficient} teilen: {answer}');steps.append(f'Probe: {a*x+b} = {c*x+d}')
    return {'expression':expression,'result':answer,'steps':steps,'engine':'bounded-linear-equation','model_generated':False}
