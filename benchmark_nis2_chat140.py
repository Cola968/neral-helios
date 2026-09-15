"""Frozen NIS2 chat-core v2 benchmark for Neral.

V2 is additive to the frozen v1 suite. It expands coverage from 40 pair cases + 8 exact
cases to 100 pair-preference cases, 24 exact-contract generation cases, and 16 open
chat-generation health probes (140 cases total). V1 remains untouched for historical
comparability.

The suite is model-only evaluation: no Helios/SOLVE/tools, retrieval, external model,
or training feedback is used.
"""
from __future__ import annotations

import hashlib
import json

SUITE_VERSION = "nis2-chat-core-v2-140"

PAIR_CASES = [
    # 1) Strict instruction following (10)
    ("instruction_following", "User: Antworte nur mit dem Wort BLAU.\nAssistant:", " BLAU", " Die Farbe ist blau."),
    ("instruction_following", "User: Reply with exactly YES and nothing else.\nAssistant:", " YES", " Yes, that is correct."),
    ("instruction_following", "User: Gib genau drei Ziffern aus: 314.\nAssistant:", " 314", " 3 1 4"),
    ("instruction_following", "User: Output only: alpha-beta\nAssistant:", " alpha-beta", " Here is alpha-beta"),
    ("instruction_following", "User: Antworte mit genau einem Wort: Berlin.\nAssistant:", " Berlin", " Berlin ist die Antwort."),
    ("instruction_following", "User: Return the lowercase word orange only.\nAssistant:", " orange", " Orange"),
    ("instruction_following", "User: Schreibe nur die Zahl 7.\nAssistant:", " 7", " Sieben"),
    ("instruction_following", "User: Answer only with FALSE.\nAssistant:", " FALSE", " The answer is FALSE."),
    ("instruction_following", "User: Gib ausschließlich OK aus.\nAssistant:", " OK", " OK!"),
    ("instruction_following", "User: Reply with one token-like word: ready\nAssistant:", " ready", " I am ready"),

    # 2) Multi-turn dialogue state (10)
    ("dialogue_state", "User: Mein Projekt heißt Komet.\nAssistant: Verstanden.\nUser: Wie heißt mein Projekt?\nAssistant:", " Komet", " Helios"),
    ("dialogue_state", "User: For this conversation, the access code is 4821.\nAssistant: Understood.\nUser: What is the access code?\nAssistant:", " 4821", " 4281"),
    ("dialogue_state", "User: Lea mag Tee, Tom mag Kaffee.\nAssistant: Okay.\nUser: Was mag Tom?\nAssistant:", " Kaffee", " Tee"),
    ("dialogue_state", "User: The red box contains a key; the blue box is empty.\nAssistant: Got it.\nUser: Which box contains the key?\nAssistant:", " The red box.", " The blue box."),
    ("dialogue_state", "User: Nenne mich in diesem Chat Nova.\nAssistant: In Ordnung.\nUser: Wie sollst du mich nennen?\nAssistant:", " Nova", " Neral"),
    ("dialogue_state", "User: Today in this fictional scenario, Mars has two stations and Venus has five.\nAssistant: Noted.\nUser: How many stations does Venus have?\nAssistant:", " Five.", " Two."),
    ("dialogue_state", "User: A=Birne, B=Apfel.\nAssistant: Verstanden.\nUser: Wofür steht B?\nAssistant:", " Apfel", " Birne"),
    ("dialogue_state", "User: First remember: the meeting room is C12.\nAssistant: Okay.\nUser: Which room did I specify?\nAssistant:", " C12", " C21"),
    ("dialogue_state", "User: Das Passwortwort ist 'Falke'.\nAssistant: Gemerkt.\nUser: Welches Wort sollst du erinnern?\nAssistant:", " Falke", " Adler"),
    ("dialogue_state", "User: In this story, Ava owns the green bike and Eli owns the black bike.\nAssistant: Understood.\nUser: Who owns the black bike?\nAssistant:", " Eli", " Ava"),

    # 3) Truthfulness / calibrated uncertainty (10)
    ("truthfulness", "User: Ist die Sonne ein Planet?\nAssistant:", " Nein, die Sonne ist ein Stern.", " Ja, die Sonne ist ein Planet."),
    ("truthfulness", "User: Do humans normally breathe underwater without equipment?\nAssistant:", " No.", " Yes."),
    ("truthfulness", "User: Wie lautet die geheime PIN, die ich dir nie genannt habe?\nAssistant:", " Das weiß ich nicht.", " Die PIN ist 1234."),
    ("truthfulness", "User: What color is the car in an image that was not provided?\nAssistant:", " I cannot know from the information provided.", " The car is blue."),
    ("truthfulness", "User: Behauptung: Berlin liegt in Frankreich. Stimmt das?\nAssistant:", " Nein, Berlin liegt in Deutschland.", " Ja, das stimmt."),
    ("truthfulness", "User: Is 1 kilogram heavier than 1,000 grams?\nAssistant:", " No, they are equal in mass.", " Yes, 1 kilogram is heavier."),
    ("truthfulness", "User: Wer gewann ein erfundenes Turnier, zu dem keine Informationen gegeben wurden?\nAssistant:", " Das lässt sich aus den Angaben nicht bestimmen.", " Berlin gewann das Turnier."),
    ("truthfulness", "User: Can you know today's lottery numbers before they are drawn?\nAssistant:", " No.", " Yes, with certainty."),
    ("truthfulness", "User: Ich habe keine Datei hochgeladen. Was steht in Zeile 12 der Datei?\nAssistant:", " Das kann ich ohne die Datei nicht wissen.", " In Zeile 12 steht Hallo."),
    ("truthfulness", "User: Is the exact weather here knowable without location or weather data?\nAssistant:", " No, not from the information given.", " Yes, it is sunny."),

    # 4) Robustness to casing/noise (10)
    ("robustness", "User: wAs iSt 2 + 3?\nAssistant:", " 5", " 6"),
    ("robustness", "User: Hauptstadt vOn DEUTSCHLAND??\nAssistant:", " Berlin", " Paris"),
    ("robustness", "User: WhAt planet do we live on???\nAssistant:", " Earth", " Mars"),
    ("robustness", "User: 7 mal 8 = ?!\nAssistant:", " 56", " 54"),
    ("robustness", "User: bittee nur kurz: wasser gefriert bei wie viel grad C?\nAssistant:", " 0", " 100"),
    ("robustness", "User: IF a room is DARK, what could u turn on?\nAssistant:", " A light.", " A sandwich."),
    ("robustness", "User: 10-3???\nAssistant:", " 7", " 8"),
    ("robustness", "User: erde kreist um... sonne oder mond??\nAssistant:", " die Sonne", " den Mond"),
    ("robustness", "User: wIe ViEl iSt 9+1????\nAssistant:", " 10", " 11"),
    ("robustness", "User: CAPITAL of france pls???\nAssistant:", " Paris", " Rome"),

    # 5) Compositional / negation reasoning (10)
    ("compositional_reasoning", "User: Alle Rivos sind blau. Teka ist ein Rivo. Welche Farbe hat Teka?\nAssistant:", " Blau.", " Rot."),
    ("compositional_reasoning", "User: Anna ist größer als Ben. Ben ist größer als Cem. Wer ist am kleinsten?\nAssistant:", " Cem.", " Anna."),
    ("compositional_reasoning", "User: If no red objects are metal and X is red, can X be metal under the rule?\nAssistant:", " No.", " Yes."),
    ("compositional_reasoning", "User: Der Schlüssel ist nicht in Box A. Er ist entweder in A oder B. Wo ist er?\nAssistant:", " In Box B.", " In Box A."),
    ("compositional_reasoning", "User: Mia is north of Leo. Leo is north of Kai. Who is farthest south?\nAssistant:", " Kai.", " Mia."),
    ("compositional_reasoning", "User: Wenn P Q impliziert und P gilt, was folgt?\nAssistant:", " Q gilt.", " Q gilt nicht."),
    ("compositional_reasoning", "User: A train stops after station B but before station D; station C is between B and D. Which comes first, C or D?\nAssistant:", " C.", " D."),
    ("compositional_reasoning", "User: Kein Lumo ist rund. Zed ist ein Lumo. Ist Zed rund?\nAssistant:", " Nein.", " Ja."),
    ("compositional_reasoning", "User: A is left of B and B is left of C. Is A left of C?\nAssistant:", " Yes.", " No."),
    ("compositional_reasoning", "User: Eva ist älter als Finn. Finn ist älter als Gina. Wer ist am ältesten?\nAssistant:", " Eva.", " Gina."),

    # 6) Arithmetic reasoning (10)
    ("arithmetic_reasoning", "User: What is 12 + 19?\nAssistant:", " 31", " 32"),
    ("arithmetic_reasoning", "User: 15 mal 4?\nAssistant:", " 60", " 45"),
    ("arithmetic_reasoning", "User: What is 81 / 9?\nAssistant:", " 9", " 8"),
    ("arithmetic_reasoning", "User: 100 minus 37?\nAssistant:", " 63", " 73"),
    ("arithmetic_reasoning", "User: If a box has 6 rows of 7 items, how many items?\nAssistant:", " 42", " 36"),
    ("arithmetic_reasoning", "User: Half of 84 is?\nAssistant:", " 42", " 44"),
    ("arithmetic_reasoning", "User: Was ist 3 hoch 3?\nAssistant:", " 27", " 9"),
    ("arithmetic_reasoning", "User: A ticket costs 8 euros. Three tickets cost?\nAssistant:", " 24 euros", " 11 euros"),
    ("arithmetic_reasoning", "User: 14 + 8 - 5 = ?\nAssistant:", " 17", " 27"),
    ("arithmetic_reasoning", "User: What is 25 percent of 80?\nAssistant:", " 20", " 25"),

    # 7) Bilingual control / transfer (10)
    ("bilingual_control", "User: Translate 'Haus' to English.\nAssistant:", " house", " horse"),
    ("bilingual_control", "User: Übersetze 'window' ins Deutsche.\nAssistant:", " Fenster", " Winter"),
    ("bilingual_control", "User: Answer in German: What is the capital of Germany?\nAssistant:", " Berlin", " Germany"),
    ("bilingual_control", "User: Antworte auf Englisch: Wie heißt die Erde auf Englisch?\nAssistant:", " Earth", " Erde"),
    ("bilingual_control", "User: Translate 'blue sky' into German.\nAssistant:", " blauer Himmel", " blue sky"),
    ("bilingual_control", "User: Übersetze 'science' ins Deutsche.\nAssistant:", " Wissenschaft", " Sprache"),
    ("bilingual_control", "User: Reply in English: Zwei plus zwei?\nAssistant:", " Four", " Vier"),
    ("bilingual_control", "User: Antworte auf Deutsch: What color is snow usually?\nAssistant:", " Weiß", " White"),
    ("bilingual_control", "User: Translate 'Flugzeug' to English.\nAssistant:", " airplane", " airport"),
    ("bilingual_control", "User: Übersetze 'planet' ins Deutsche.\nAssistant:", " Planet", " Pflanze"),

    # 8) Format control (10)
    ("format_control", "User: Answer as exactly two comma-separated words: red,blue\nAssistant:", " red,blue", " red and blue"),
    ("format_control", "User: Gib die Antwort in Klammern aus: (42)\nAssistant:", " (42)", " 42"),
    ("format_control", "User: Output lowercase only: berlin\nAssistant:", " berlin", " Berlin"),
    ("format_control", "User: Reply with the exact JSON literal {\"ok\":true}.\nAssistant:", " {\"ok\":true}", " ok=true"),
    ("format_control", "User: Gib genau A-B-C aus.\nAssistant:", " A-B-C", " A B C"),
    ("format_control", "User: Output exactly one digit: 5\nAssistant:", " 5", " five"),
    ("format_control", "User: Reply with uppercase HELLO only.\nAssistant:", " HELLO", " Hello"),
    ("format_control", "User: Gib exakt 'ja/nein' aus.\nAssistant:", " ja/nein", " ja oder nein"),
    ("format_control", "User: Output exactly: x_y\nAssistant:", " x_y", " x-y"),
    ("format_control", "User: Reply with a single period.\nAssistant:", " .", " ..."),

    # 9) Consistency / contradiction handling (10)
    ("consistency", "User: The rule says all Zips are green. A is a Zip. Is A green?\nAssistant:", " Yes.", " No."),
    ("consistency", "User: I just said the door is closed. Is the door open according to my statement?\nAssistant:", " No.", " Yes."),
    ("consistency", "User: Aussage: 5 ist größer als 3. Ist das wahr?\nAssistant:", " Ja.", " Nein."),
    ("consistency", "User: If the premise is 'no birds are mammals', does it imply birds are mammals?\nAssistant:", " No.", " Yes."),
    ("consistency", "User: Ich sage: Der Ball ist rot, nicht blau. Welche Farbe habe ich genannt?\nAssistant:", " Rot.", " Blau."),
    ("consistency", "User: Fact for this question: X=9. What is X?\nAssistant:", " 9", " 8"),
    ("consistency", "User: The text says the train arrived Monday, not Tuesday. Which day?\nAssistant:", " Monday", " Tuesday"),
    ("consistency", "User: Regel: A bedeutet links. Was bedeutet A?\nAssistant:", " links", " rechts"),
    ("consistency", "User: Assume in this puzzle that water is purple. What color is water in the puzzle?\nAssistant:", " Purple.", " Blue."),
    ("consistency", "User: In this fictional setting, 2+2 is defined as 5. Under that setting, what is 2+2?\nAssistant:", " 5", " 4"),

    # 10) Context binding / distractor resistance (10)
    ("context_binding", "User: Ria chose bus 17. Niko chose bus 24. Which bus did Ria choose?\nAssistant:", " 17", " 24"),
    ("context_binding", "User: The apple is on shelf 2; the pear is on shelf 5. Where is the pear?\nAssistant:", " Shelf 5.", " Shelf 2."),
    ("context_binding", "User: Berlin=red, Paris=blue for this puzzle. What color is Paris?\nAssistant:", " blue", " red"),
    ("context_binding", "User: K1 means north; K2 means south. What does K2 mean?\nAssistant:", " south", " north"),
    ("context_binding", "User: Lara scored 8, Miro scored 6, Tia scored 9. Who scored 6?\nAssistant:", " Miro", " Tia"),
    ("context_binding", "User: Box A has 3 coins and Box B has 7. How many are in Box A?\nAssistant:", " 3", " 7"),
    ("context_binding", "User: Alpha arrives first, Beta second, Gamma third. Who is second?\nAssistant:", " Beta", " Gamma"),
    ("context_binding", "User: Der Hund heißt Milo, die Katze heißt Luna. Wie heißt die Katze?\nAssistant:", " Luna", " Milo"),
    ("context_binding", "User: Route X ends in Oslo; route Y ends in Rome. Where does route X end?\nAssistant:", " Oslo", " Rome"),
    ("context_binding", "User: Code A7 belongs to Nova; code B4 belongs to Vega. Which code belongs to Vega?\nAssistant:", " B4", " A7"),
]

EXACT_GENERATION_CASES = [
    ("if-exact-01", "User: Antworte exakt mit: BLAU\nAssistant:", ["BLAU"]),
    ("if-exact-02", "User: Reply exactly with: YES\nAssistant:", ["YES"]),
    ("if-exact-03", "User: Gib nur diese Ziffern aus: 314\nAssistant:", ["314"]),
    ("if-exact-04", "User: Output exactly: alpha-beta\nAssistant:", ["alpha-beta"]),
    ("if-exact-05", "User: Antworte nur mit: 7\nAssistant:", ["7"]),
    ("if-exact-06", "User: Return exactly this lowercase word: orange\nAssistant:", ["orange"]),
    ("if-exact-07", "User: Antworte exakt mit: OK\nAssistant:", ["OK"]),
    ("if-exact-08", "User: Reply with exactly: FALSE\nAssistant:", ["FALSE"]),
    ("if-exact-09", "User: Output only: berlin\nAssistant:", ["berlin"]),
    ("if-exact-10", "User: Gib exakt aus: A-B-C\nAssistant:", ["A-B-C"]),
    ("if-exact-11", "User: Reply exactly: ready\nAssistant:", ["ready"]),
    ("if-exact-12", "User: Output exactly: x_y\nAssistant:", ["x_y"]),
    ("if-exact-13", "User: Antworte nur: 42\nAssistant:", ["42"]),
    ("if-exact-14", "User: Reply only: NO\nAssistant:", ["NO"]),
    ("if-exact-15", "User: Gib nur aus: grün\nAssistant:", ["grün", "gruen"]),
    ("if-exact-16", "User: Output exactly: 3.14\nAssistant:", ["3.14"]),
    ("if-exact-17", "User: Antworte exakt: Berlin\nAssistant:", ["Berlin"]),
    ("if-exact-18", "User: Reply exactly: Earth\nAssistant:", ["Earth"]),
    ("if-exact-19", "User: Output only: 0\nAssistant:", ["0"]),
    ("if-exact-20", "User: Gib exakt aus: ja/nein\nAssistant:", ["ja/nein"]),
    ("if-exact-21", "User: Reply exactly: HELLO\nAssistant:", ["HELLO"]),
    ("if-exact-22", "User: Output only: 12345\nAssistant:", ["12345"]),
    ("if-exact-23", "User: Antworte exakt mit einem Punkt: .\nAssistant:", ["."]),
    ("if-exact-24", "User: Reply exactly: done\nAssistant:", ["done"]),
]

# Open-ended probes are not accuracy-scored. They measure termination and degeneration.
GENERATION_HEALTH_CASES = [
    ("gh-01", "User: Nenne eine Farbe.\nAssistant:"),
    ("gh-02", "User: Name one planet.\nAssistant:"),
    ("gh-03", "User: Was ist 2+2?\nAssistant:"),
    ("gh-04", "User: Complete briefly: Berlin is a city in\nAssistant:"),
    ("gh-05", "User: Say hello in German.\nAssistant:"),
    ("gh-06", "User: Nenne ein Tier.\nAssistant:"),
    ("gh-07", "User: What is the opposite of hot?\nAssistant:"),
    ("gh-08", "User: Schreibe einen kurzen Satz über den Mond.\nAssistant:"),
    ("gh-09", "User: Give one programming language.\nAssistant:"),
    ("gh-10", "User: Was ist die Hauptstadt von Frankreich?\nAssistant:"),
    ("gh-11", "User: Answer briefly: Is water wet?\nAssistant:"),
    ("gh-12", "User: Nenne eine gerade Zahl.\nAssistant:"),
    ("gh-13", "User: Name a common fruit.\nAssistant:"),
    ("gh-14", "User: Ergänze kurz: Die Erde kreist um die\nAssistant:"),
    ("gh-15", "User: What comes after Monday?\nAssistant:"),
    ("gh-16", "User: Sage kurz guten Morgen.\nAssistant:"),
]


def suite_sha256() -> str:
    payload = json.dumps(
        {
            "version": SUITE_VERSION,
            "pairs": PAIR_CASES,
            "exact": EXACT_GENERATION_CASES,
            "generation_health": GENERATION_HEALTH_CASES,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_suite() -> None:
    if len(PAIR_CASES) != 100:
        raise RuntimeError(f"expected 100 pair cases, got {len(PAIR_CASES)}")
    if len(EXACT_GENERATION_CASES) != 24:
        raise RuntimeError(f"expected 24 exact cases, got {len(EXACT_GENERATION_CASES)}")
    if len(GENERATION_HEALTH_CASES) != 16:
        raise RuntimeError(f"expected 16 generation-health cases, got {len(GENERATION_HEALTH_CASES)}")
    categories = {}
    for category, *_ in PAIR_CASES:
        categories[category] = categories.get(category, 0) + 1
    if len(categories) != 10 or any(v != 10 for v in categories.values()):
        raise RuntimeError(f"expected 10x10 pair category balance, got {categories}")
    exact_ids = [x[0] for x in EXACT_GENERATION_CASES]
    health_ids = [x[0] for x in GENERATION_HEALTH_CASES]
    if len(set(exact_ids)) != len(exact_ids) or len(set(health_ids)) != len(health_ids):
        raise RuntimeError("duplicate NIS2 case id")


validate_suite()
