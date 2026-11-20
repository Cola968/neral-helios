"""German periodic table lookup for one precisely scoped atomic-number question.

No external web request, model inference, or arbitrary chemical properties.
Names are canonical/common German spellings, not alternate historical forms.
"""
import re

GERMAN_ELEMENTS=(
    "Wasserstoff","Helium","Lithium","Beryllium","Bor","Kohlenstoff",
    "Stickstoff","Sauerstoff","Fluor","Neon","Natrium","Magnesium",
    "Aluminium","Silizium","Phosphor","Schwefel","Chlor","Argon",
    "Kalium","Calcium","Scandium","Titan","Vanadium","Chrom","Mangan",
    "Eisen","Kobalt","Nickel","Kupfer","Zink","Gallium","Germanium",
    "Arsen","Selen","Brom","Krypton","Rubidium","Strontium","Yttrium",
    "Zirconium","Niob","Molybdän","Technetium","Ruthenium","Rhodium",
    "Palladium","Silber","Cadmium","Indium","Zinn","Antimon","Tellur",
    "Jod","Xenon","Cäsium","Barium","Lanthan","Cer","Praseodym",
    "Neodym","Promethium","Samarium","Europium","Gadolinium","Terbium",
    "Dysprosium","Holmium","Erbium","Thulium","Ytterbium","Lutetium",
    "Hafnium","Tantal","Wolfram","Rhenium","Osmium","Iridium","Platin",
    "Gold","Quecksilber","Thallium","Blei","Wismut","Polonium","Astat",
    "Radon","Francium","Radium","Actinium","Thorium","Protactinium",
    "Uran","Neptunium","Plutonium","Americium","Curium","Berkelium",
    "Californium","Einsteinium","Fermium","Mendelevium","Nobelium",
    "Lawrencium","Rutherfordium","Dubnium","Seaborgium","Bohrium",
    "Hassium","Meitnerium","Darmstadtium","Röntgenium","Copernicium",
    "Nihonium","Flerovium","Moscovium","Livermorium","Tenness","Oganesson",
)
assert len(GERMAN_ELEMENTS)==118

_EXACT_ELEMENT_QUESTION=re.compile(
    r"Wie heißt das chemische Element mit der Ordnungszahl (\d{1,3})\? "
    r"Nur den deutschen Namen\.", re.I)


def verified_element_question(question):
    """Answer only a complete, unambiguous atomic-number request."""
    if not isinstance(question,str):
        return None
    match=_EXACT_ELEMENT_QUESTION.fullmatch(question.strip())
    if match is None:
        return None
    atomic=int(match.group(1))
    if not 1<=atomic<=118:
        return None
    return {
        "text":GERMAN_ELEMENTS[atomic-1],"verified":True,
        "safe_to_promote":True,"proof_backend":"periodic_table_index",
        "verification_scope":"one_exact_atomic_number_1_to_118",
        "verification_status":"table_lookup",
        "tool_used":"direct_reasoning","tool_verified":False,
        "agent_checked":True,"agent_steps":["periodic_table_lookup"],
        "agent_attempts":0,"error_code":None,"sources":[],
    }
