from __future__ import annotations

"""Larger development-only diagnostic for Arctus recovery work.

This suite is deliberately separate from the frozen final audit. It exists to reduce the
noise of the tiny 8-item/category diagnostic while tuning curriculum/LR. Never include
these cases in training data and never use them as a public benchmark claim.
"""

import argparse
import collections
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model350 import FRONTIER_350, Neral350M
from eval500m_diagnostic import conditional_mean_nll

SUITE_VERSION = "arctus-recovery-dev-v1"

CASES: list[tuple[str, str, str, str, list[str]]] = []

def add(cat: str, prefix: str, preferred: str, distractors: list[str]) -> None:
    CASES.append((f"{cat}-{len([x for x in CASES if x[1] == cat])+1:02d}", cat, prefix, preferred, distractors))

# Factual knowledge: stable, simple facts with mixed DE/EN phrasing.
factual = [
    ("The capital of Italy is", " Rome.", [" Milan.", " Paris.", " Madrid."]),
    ("The capital of Spain is", " Madrid.", [" Barcelona.", " Rome.", " Lisbon."]),
    ("The capital of Austria is", " Vienna.", [" Salzburg.", " Berlin.", " Prague."]),
    ("The capital of Poland is", " Warsaw.", [" Krakow.", " Berlin.", " Prague."]),
    ("The capital of Japan is", " Tokyo.", [" Kyoto.", " Seoul.", " Beijing."]),
    ("The largest ocean on Earth is the", " Pacific Ocean.", [" Atlantic Ocean.", " Indian Ocean.", " Arctic Ocean."]),
    ("The chemical symbol for gold is", " Au.", [" Ag.", " Fe.", " Gd."]),
    ("The chemical symbol for iron is", " Fe.", [" Ir.", " In.", " I."]),
    ("The Earth has", " one Moon.", [" two Moons.", " no Moon.", " ten Moons."]),
    ("A week has", " seven days.", [" five days.", " six days.", " ten days."]),
    ("Die Hauptstadt von Frankreich ist", " Paris.", [" Lyon.", " Madrid.", " Rom."]),
    ("Die Hauptstadt von Österreich ist", " Wien.", [" Graz.", " Berlin.", " Prag."]),
    ("Deutschland liegt in", " Europa.", [" Asien.", " Afrika.", " Südamerika."]),
    ("Der höchste Berg der Erde ist", " der Mount Everest.", [" der Brocken.", " der Kilimandscharo.", " der Mont Blanc."]),
    ("The planet closest to the Sun is", " Mercury.", [" Venus.", " Earth.", " Mars."]),
    ("The human heart pumps", " blood.", [" air.", " water.", " bile."]),
    ("Water is made of hydrogen and", " oxygen.", [" carbon.", " iron.", " helium."]),
    ("The opposite direction of north is", " south.", [" east.", " west.", " up."]),
    ("The currency used in Germany is the", " euro.", [" dollar.", " pound.", " yen."]),
    ("Berlin is a city in", " Germany.", [" France.", " Italy.", " Spain."]),
]
for x in factual: add("factual", *x)

# Math: exact elementary arithmetic/algebra to expose genuine transfer rather than prose priors.
math_cases = [
    ("3 + 4 =", " 7", [" 6", " 8", " 12"]), ("8 + 5 =", " 13", [" 12", " 14", " 40"]),
    ("15 - 6 =", " 9", [" 8", " 10", " 21"]), ("20 - 7 =", " 13", [" 12", " 14", " 27"]),
    ("5 * 6 =", " 30", [" 25", " 35", " 56"]), ("9 * 4 =", " 36", [" 32", " 40", " 94"]),
    ("18 / 3 =", " 6", [" 3", " 9", " 15"]), ("24 / 6 =", " 4", [" 3", " 5", " 18"]),
    ("The square root of 64 is", " 8.", [" 6.", " 7.", " 64."]),
    ("The square root of 49 is", " 7.", [" 6.", " 8.", " 49."]),
    ("If x + 3 = 10, then x =", " 7.", [" 3.", " 10.", " 13."]),
    ("If 2x = 14, then x =", " 7.", [" 6.", " 12.", " 28."]),
    ("50 percent of 80 is", " 40.", [" 20.", " 50.", " 80."]),
    ("10 percent of 90 is", " 9.", [" 10.", " 18.", " 81."]),
    ("3 squared is", " 9.", [" 6.", " 3.", " 27."]),
    ("2 cubed is", " 8.", [" 4.", " 6.", " 16."]),
    ("12 plus 8 equals", " 20.", [" 18.", " 21.", " 96."]),
    ("30 minus 12 equals", " 18.", [" 16.", " 20.", " 42."]),
    ("6 mal 9 ist", " 54.", [" 45.", " 56.", " 69."]),
    ("100 geteilt durch 4 ist", " 25.", [" 20.", " 40.", " 96."]),
]
for x in math_cases: add("math", *x)

science = [
    ("The center of an atom contains protons and", " neutrons.", [" photons.", " planets.", " cells."]),
    ("Plants absorb carbon dioxide and release", " oxygen.", [" nitrogen.", " helium.", " iron."]),
    ("The SI unit of force is the", " newton.", [" watt.", " volt.", " meter."]),
    ("The SI unit of electric current is the", " ampere.", [" joule.", " kelvin.", " pascal."]),
    ("A solid changing directly into a gas is called", " sublimation.", [" condensation.", " freezing.", " ionization."]),
    ("Earth's natural satellite is the", " Moon.", [" Sun.", " Mars.", " Venus."]),
    ("Jupiter is primarily a", " gas giant.", [" rocky moon.", " dwarf planet.", " star."]),
    ("The Sun is a", " star.", [" planet.", " moon.", " galaxy."]),
    ("Light travels fastest in", " vacuum.", [" water.", " glass.", " steel."]),
    ("Sound requires a", " medium to propagate.", [" vacuum to propagate.", " magnetic field only.", " star to propagate."]),
    ("DNA carries", " genetic information.", [" electric current.", " sound waves.", " gravity."]),
    ("Mitochondria are found inside", " cells.", [" planets.", " atoms.", " oceans."]),
    ("A proton has", " positive electric charge.", [" negative electric charge.", " no charge.", " two negative charges."]),
    ("An electron has", " negative electric charge.", [" positive electric charge.", " no charge.", " two positive charges."]),
    ("Acceleration due to gravity near Earth's surface is about", " 9.8 m/s².", [" 0.98 m/s².", " 98 m/s².", " 300000 m/s²."]),
    ("The event horizon surrounds a", " black hole.", [" red giant only.", " comet.", " cell nucleus."]),
    ("Die Erde umkreist", " die Sonne.", [" den Mars.", " den Mond.", " den Jupiter."]),
    ("Der Mond umkreist", " die Erde.", [" die Sonne allein.", " den Mars.", " die Venus."]),
    ("Wasser siedet bei Normaldruck ungefähr bei", " 100 Grad Celsius.", [" 0 Grad Celsius.", " 20 Grad Celsius.", " 500 Grad Celsius."]),
    ("Ein Photon ist ein Quant des", " elektromagnetischen Feldes.", [" Gravitationsfeldes ausschließlich.", " Blutkreislaufs.", " Zellkerns."]),
]
for x in science: add("science", *x)

reasoning = [
    ("All roses are flowers. This is a rose. Therefore it is", " a flower.", [" a planet.", " not a flower.", " a mineral."]),
    ("All squares are rectangles. This shape is a square. Therefore it is", " a rectangle.", [" a circle.", " not a rectangle.", " a triangle only."]),
    ("Mia is taller than Leo. Leo is taller than Kim. Therefore Mia is", " taller than Kim.", [" shorter than Kim.", " the same height as Kim.", " unrelated to Kim."]),
    ("A box has 4 red balls and 3 blue balls. It has", " 7 balls total.", [" 1 ball total.", " 12 balls total.", " 4 balls total."]),
    ("A movie starts at 18:00 and lasts two hours. It ends at", " 20:00.", [" 16:00.", " 18:30.", " 22:00."]),
    ("If every X is Y and every Y is Z, every X is", " Z.", [" not Z.", " only Y.", " unrelated to Z."]),
    ("No fish are birds. A salmon is a fish. Therefore a salmon is", " not a bird.", [" a bird.", " a mammal by necessity.", " a tree."]),
    ("If the lamp is on, the room is lit. The lamp is on. Therefore the room is", " lit.", [" dark by necessity.", " underwater.", " frozen."]),
    ("Sam has 10 euros and spends 3 euros. Sam has", " 7 euros left.", [" 13 euros left.", " 3 euros left.", " 30 euros left."]),
    ("There are 5 books on a shelf and 2 are removed. There are", " 3 books left.", [" 7 books left.", " 2 books left.", " 10 books left."]),
    ("If today is Friday, tomorrow is", " Saturday.", [" Thursday.", " Monday.", " Friday."]),
    ("If yesterday was Sunday, today is", " Monday.", [" Saturday.", " Tuesday.", " Sunday."]),
    ("A is left of B. B is left of C. Therefore A is", " left of C.", [" right of C.", " equal to C.", " above C."]),
    ("Nina is younger than Omar. Therefore Omar is", " older than Nina.", [" younger than Nina.", " the same age as Nina.", " unrelated to Nina."]),
    ("A train travels 60 km in one hour at constant speed. In two hours it travels", " 120 km.", [" 30 km.", " 60 km.", " 180 km."]),
    ("Three identical bags contain 2 apples each. Together they contain", " 6 apples.", [" 3 apples.", " 5 apples.", " 8 apples."]),
    ("Wenn heute Mittwoch ist, ist morgen", " Donnerstag.", [" Dienstag.", " Montag.", " Mittwoch."]),
    ("Paul ist älter als Eva. Eva ist älter als Tom. Also ist Paul", " älter als Tom.", [" jünger als Tom.", " gleich alt wie Tom.", " nicht vergleichbar."]),
    ("Alle Hunde sind Tiere. Bello ist ein Hund. Also ist Bello", " ein Tier.", [" eine Pflanze.", " kein Tier.", " ein Planet."]),
    ("Ein Bus fährt um 14:00 ab und braucht eine Stunde. Er kommt um", " 15:00 an.", [" 13:00 an.", " 14:30 an.", " 16:00 an."]),
]
for x in reasoning: add("reasoning", *x)

language = [
    ("He ___ coffee every morning.", " drinks", [" drink", " drinking", " drank tomorrow"]),
    ("They ___ to Berlin yesterday.", " went", [" go", " goes", " will go yesterday"]),
    ("The plural of mouse is", " mice.", [" mouses.", " mousees.", " mouse."]),
    ("The plural of person is", " people.", [" personses.", " persones.", " person."]),
    ("A synonym of quick is", " fast.", [" slow.", " heavy.", " dark."]),
    ("The opposite of early is", " late.", [" soon.", " quick.", " near."]),
    ("She is ___ than her brother.", " taller", [" tallest", " talling", " more tallly"]),
    ("I have ___ the book.", " read", [" reading", " reads", " rode"]),
    ("Ich ___ heute zur Schule.", " gehe", [" geht", " gehen du", " ging morgen"]),
    ("Wir ___ gestern im Park.", " waren", [" war", " seid", " ist"]),
    ("Der Plural von Kind ist", " Kinder.", [" Kinds.", " Kinderns.", " Kindes."]),
    ("Der Plural von Buch ist", " Bücher.", [" Buchs.", " Bucher.", " Büchernis."]),
    ("Das Gegenteil von kalt ist", " warm.", [" langsam.", " dunkel.", " klein."]),
    ("Ein Synonym für beginnen ist", " anfangen.", [" enden.", " stoppen.", " vergessen."]),
    ("Der Hund ___ im Garten.", " spielt", [" spielen", " spielst ich", " gespielt morgen"]),
    ("Gestern ___ es stark.", " regnete", [" regnet morgen", " regnen", " regnest"]),
    ("A sentence normally begins with a", " capital letter.", [" comma.", " footnote.", " number only."]),
    ("The past tense of 'go' is", " went.", [" goed.", " goes.", " going."]),
    ("The comparative of 'good' is", " better.", [" gooder.", " bestest.", " more goodly."]),
    ("Das Präteritum von 'sein' in 'ich' lautet", " war.", [" bin.", " ist.", " werde."]),
]
for x in language: add("language", *x)

bilingual = [
    ("The German translation of 'car' is", " Auto.", [" Baum.", " Wasser.", " Buch."]),
    ("The German translation of 'tree' is", " Baum.", [" Auto.", " Haus.", " Feuer."]),
    ("The German translation of 'school' is", " Schule.", [" Straße.", " Sonne.", " Tisch."]),
    ("The German translation of 'sun' is", " Sonne.", [" Mond.", " Stern.", " Regen."]),
    ("The German translation of 'moon' is", " Mond.", [" Sonne.", " Erde.", " Luft."]),
    ("The German translation of 'city' is", " Stadt.", [" Land.", " Dorfplatz nur.", " Straße."]),
    ("The German translation of 'friend' is", " Freund.", [" Feind.", " Vater.", " Lehrer."]),
    ("The German translation of 'red' is", " rot.", [" blau.", " grün.", " schwarz."]),
    ("The German translation of 'small' is", " klein.", [" groß.", " laut.", " schnell."]),
    ("The German translation of 'thank you' is", " danke.", [" bitte gehen.", " guten Morgen.", " nein."]),
    ("Die englische Übersetzung von 'Haus' ist", " house.", [" tree.", " road.", " book."]),
    ("Die englische Übersetzung von 'Straße' ist", " street.", [" star.", " school.", " chair."]),
    ("Die englische Übersetzung von 'Mond' ist", " moon.", [" sun.", " earth.", " cloud."]),
    ("Die englische Übersetzung von 'Freund' ist", " friend.", [" enemy.", " father.", " teacher."]),
    ("Die englische Übersetzung von 'rot' ist", " red.", [" blue.", " green.", " black."]),
    ("Die englische Übersetzung von 'klein' ist", " small.", [" large.", " fast.", " loud."]),
    ("Deutsch: 'Guten Abend.' English:", " Good evening.", [" Good morning.", " Good night only.", " Thank you."]),
    ("English: 'See you tomorrow.' German:", " Bis morgen.", [" Guten Morgen.", " Gute Nacht.", " Danke sehr."]),
    ("Deutsch: 'Ich bin müde.' English:", " I am tired.", [" I am hungry.", " I am late.", " I am outside."]),
    ("English: 'Where is the station?' German:", " Wo ist der Bahnhof?", [" Wie spät ist es?", " Wo ist die Schule?", " Wer bist du?"]),
]
for x in bilingual: add("bilingual", *x)

code = [
    ("def inc(x):\n    return", " x + 1", [" x - 1", " 0", " None"]),
    ("def multiply(a, b):\n    return", " a * b", [" a + b", " a / 0", " None"]),
    ("def is_positive(x):\n    return x >", " 0", [" 1000000", " x", " None"]),
    ("nums = [1, 2, 3]\nfirst = ", "nums[0]", ["nums[3]", "len(nums)", "sum(nums)"]),
    ("items = []\nitems.", "append(1)", ["length(1)", "push = None", "remove_all(1)"]),
    ("for i in range(5):\n    ", "print(i)", ["print(range)", "return i", "break i"]),
    ("def max2(a, b):\n    return a if a > b else", " b", [" a", " None", " False"]),
    ("x = 5\nif x == 5:\n    ", "print(\"yes\")", ["print(\"no\")", "x = None", "raise 5"]),
    ("const double = x =>", " x * 2;", [" x / 0;", " null;", " x === 2;"]),
    ("SELECT name FROM users WHERE id =", " 1;", [" users;", " SELECT;", " name;"]),
    ("def length(xs):\n    return", " len(xs)", [" sum(xs)", " xs[999]", " range(xs)"]),
    ("def negate(x):\n    return", " -x", [" x", " +x + x", " None"]),
    ("a = 3\nb = 4\nc =", " a + b", [" a - b", " a == b", " None"]),
    ("def odd(n):\n    return n % 2 ==", " 1", [" 0", " 2", " n"]),
    ("for item in items:\n    print(", "item)", ["items)", "range)", "None)"]),
    ("let x = 2;\nlet y =", " x + 1;", [" x / 0;", " null;", " false;"]),
    ("def empty(xs):\n    return len(xs) ==", " 0", [" 1", " xs", " None"]),
    ("value = True\nresult = not", " value", [" True value", " None", " 0 value"]),
    ("def identity(x):\n    return", " x", [" None", " 0", " x + 1"]),
    ("numbers = [2, 4, 6]\ntotal =", " sum(numbers)", [" len(numbers)", " numbers[9]", " range(numbers)"]),
]
for x in code: add("code", *x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-tokens", type=int, default=1_000_000_000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    processed = int(ck.get("processed_tokens", 0))
    if processed < args.expected_tokens:
        raise RuntimeError(f"checkpoint too early: {processed} < {args.expected_tokens}")
    model = Neral350M().to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.load_state_dict(ck["model"])
    model.eval()
    bos_id = tok.token_to_id("<bos>")
    if bos_id is None:
        bos_id = tok.token_to_id("<s>")
    if bos_id is None:
        raise RuntimeError("BOS token not found")

    by_cat: dict[str, list[bool]] = collections.defaultdict(list)
    rows = []
    for case_id, cat, prefix, preferred, distractors in CASES:
        options = [preferred, *distractors]
        losses = [conditional_mean_nll(model, tok, prefix, x, device, bos_id) for x in options]
        chosen = min(range(len(losses)), key=losses.__getitem__)
        ok = chosen == 0
        by_cat[cat].append(ok)
        rows.append({"id": case_id, "category": cat, "correct": ok, "preferred": preferred, "chosen": options[chosen], "losses": losses, "margin": min(losses[1:]) - losses[0]})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    categories = {k: {"correct": sum(v), "total": len(v), "accuracy_pct": 100.0 * sum(v) / len(v)} for k, v in sorted(by_cat.items())}
    total_correct = sum(x["correct"] for x in rows)
    summary = {
        "suite": SUITE_VERSION,
        "checkpoint_processed_tokens": processed,
        "architecture": FRONTIER_350.to_dict(),
        "cases": len(rows),
        "overall_accuracy_pct": 100.0 * total_correct / len(rows),
        "categories": categories,
        "purpose": "development-only recovery diagnostic; not final audit and not a public benchmark",
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
