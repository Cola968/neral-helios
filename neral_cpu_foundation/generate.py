"""Deterministic German tasks; generated from structured specs, solved on CPU."""
from __future__ import annotations
import json
from pathlib import Path
from .symbolic import verify_symbolic
from .graphs import verify_graph
from .constraints import solve_constraints


def generate_examples(count:int=256)->list[dict]:
    if type(count) is not int or not 250<=count<=400: raise ValueError("count must be 250..400")
    result=[]
    endings=(" Antworte mit dem exakten Ergebnis."," Gib die überprüfte Lösung an."," Prüfe deine Antwort rechnerisch."," Ermittle die korrekte Lösung.")
    for i in range(count):
        group=i%4;num=i//4
        if group==0:
            a=2+num
            b=2*a+num%13+1
            if num%2==0:
                spec={"operation":"equation","equation":f"x+{a}={b}","variables":["x"]}
                prompt=f"Welche Zahl x erfüllt x + {a} = {b}?"
                variant="algebra_equation"
            else:
                spec={"operation":"differentiate","expression":f"x**2+{a}*x+{b}"}
                prompt=f"Leite die Funktion f(x)=x²+{a}x+{b} nach x ab."
                variant="algebra_derivative"
            tool="sympy"
        elif group==1:
            a,b,c=2+num,3+(num*3)%31,4+(num*7)%23
            spec={"operation":"shortest_path","directed":True,"nodes":["A","B","C","D"],
              "edges":[["A","B",a],["B","D",b],["A","C",c],["C","D",a+3],["A","D",a+b+5]],
              "source":"A","target":"D"}
            prompt=f"Ein gerichtetes Netz hat die gewichteten Kanten A→B:{a}, B→D:{b}, A→C:{c}, C→D:{a+3} und A→D:{a+b+5}. Wie lang ist der kürzeste Weg von A nach D?"
            tool="networkx";variant="graph_shortest"
        elif group==2:
            n=8+num
            k=2+(num%6)
            spec={"operation":"combination","n":n,"k":k}
            prompt=f"Aus {n} verschiedenen Elementen werden {k} ohne Reihenfolge ausgewählt. Wie viele Auswahlmöglichkeiten gibt es?"
            tool="sympy";variant="combination_choose"
        else:
            preferred=num%4
            slots=6+(num//20)
            spec={"template":"schedule","slots":slots,"tasks":[
              {"id":"A","duration":2,"resource":"R","window":[0,3],"preferred":preferred},
              {"id":"B","duration":1,"resource":"R","window":[1,slots-1],"preferred":num%5},
              {"id":"C","duration":1,"resource":"S","window":[0,slots-1],"preferred":(num+2)%6},
            ]}
            prompt=f"Plane A (2 Zeiteinheiten) und B (1) auf Ressource R ohne Überlappung sowie C (1) auf S in {slots} Slots. A darf bei 0–3, B bei 1–5 starten. Bevorzugte Startzeiten: A={preferred}, B={num%5}, C={(num+2)%6}. Minimiere Abweichungen."
            tool="z3";variant="schedule_preferences"
        try:
            answer=verify_symbolic(spec) if tool=="sympy" else verify_graph(spec) if tool=="networkx" else solve_constraints(spec)
            if not answer.get("verified"):continue
            row={"prompt":prompt+endings[num%4],"task":{"tool":tool,"spec":spec},
               "answer":answer["result"],"group":f"{variant}_{num%8}"}
            result.append(row)
        except (KeyError,ValueError,RuntimeError):continue
    return result


def save_generated(path:str|Path,count:int=256)->int:
    output=generate_examples(count)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in output),encoding="utf-8")
    return len(output)
