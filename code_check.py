"""Static Python checks only. Submitted code is never executed or imported."""
import ast
def check_code(source):
    if len(source.encode('utf-8'))>4096:raise ValueError('Maximal 4096 UTF-8-Bytes Code erlaubt.')
    try:tree=ast.parse(source)
    except SyntaxError as e:return {'syntax_ok':False,'line':e.lineno,'column':e.offset,'message':e.msg,'findings':[],'executed':False}
    except (RecursionError,ValueError) as e:raise ValueError('Code kann nicht sicher geparst werden.') from e
    findings=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ExceptHandler) and node.type is None:findings.append({'line':node.lineno,'message':'Nacktes except: möglichst konkrete Ausnahmetypen behandeln.'})
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if any(isinstance(x,(ast.List,ast.Dict,ast.Set)) for x in node.args.defaults+node.args.kw_defaults if x is not None):findings.append({'line':node.lineno,'message':'Veränderbarer Standardwert wird zwischen Aufrufen geteilt. Nutze None und initialisiere im Funktionskörper.'})
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'eval','exec'}:findings.append({'line':node.lineno,'message':'Dynamische Codeausführung: keine nicht vertrauenswürdigen Eingaben ausführen.'})
    return {'syntax_ok':True,'findings':findings,'executed':False,'note':'Syntax und ausgewählte Muster geprüft. Keine Aussage über funktionale Korrektheit oder vollständige Sicherheit.'}
