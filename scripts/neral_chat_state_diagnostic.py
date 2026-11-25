"""Offline synthetic diagnostic: can an answer preserve and revise task state?"""
import argparse
import hashlib
import json
import sys

ARMS = ("raw_qwen35", "neral")
KEYS = {"place", "slot", "bring"}
INSTRUCTION = (
    "Track the latest plan across these turns. Return only a JSON object with "
    "place and slot (strings, or null if unknown or cleared) and bring "
    "(an array of unique items still to bring). Preserve unchanged details. "
    "If a detail was cleared, do not repeat it."
)
CASES = {
    "correct_place": (
        ("Meet at the library at 10:00. Bring a notebook.",
         "Change the location to the museum. Everything else stays the same."),
        {"place": "museum", "slot": "10:00", "bring": ["notebook"]},
    ),
    "clear_slot": (
        ("Meet at the park at 15:00. Bring a kite.",
         "The time is no longer set.",
         "Add a bottle of water to the list."),
        {"place": "park", "slot": None, "bring": ["kite", "bottle of water"]},
    ),
    "remove_and_readd": (
        ("Meet at the station at 18:00. Bring a map and a torch.",
         "Don't bring the map. Change the meeting time to 19:30.",
         "We do need the map after all, but leave the torch at home."),
        {"place": "station", "slot": "19:30", "bring": ["map"]},
    ),
    "unknown_place": (
        ("The time is 09:00. No venue yet. Bring crayons.",
         "Change the time to 09:45 and remove the crayons."),
        {"place": None, "slot": "09:45", "bring": []},
    ),
    "replace_items": (
        ("Meet at the pool at 11:00. Bring sandals and a towel.",
         "New list: a swim cap only. Leave the time and location as they were."),
        {"place": "pool", "slot": "11:00", "bring": ["swim cap"]},
    ),
    "latest_update": (
        ("Meet at the cafe at 08:00. Bring a thermos.",
         "Let's meet at the bakery instead.",
         "Actually move the venue to the square and clear the time."),
        {"place": "square", "slot": None, "bring": ["thermos"]},
    ),
}


class DiagnosticError(ValueError):
    """A sanitised report error; never include raw model output."""


def _unique_object(pairs):
    row = {}
    for key, value in pairs:
        if key in row:
            raise DiagnosticError("Duplicate JSON key")
        row[key] = value
    return row


def _decode(text):
    return json.loads(text, object_pairs_hook=_unique_object)


def prompts():
    """Expose only the input messages, not their scoring references."""
    rows = {}
    for case_id, (turns, _) in CASES.items():
        messages = [{"role": "system", "content": INSTRUCTION}]
        for turn in turns[:-1]:
            messages.extend(({"role": "user", "content": turn},
                             {"role": "assistant", "content": "Noted."}))
        messages.append({"role": "user", "content": turns[-1]})
        wire = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        rows[case_id] = {"messages": messages,
                         "transcript_sha256": hashlib.sha256(wire.encode("utf8")).hexdigest()}
    return rows


def _matches(answer, reference):
    if type(answer) is not str or len(answer) > 2048:
        return False
    try:
        value = _decode(answer)
    except (ValueError, RecursionError):
        return False
    if type(value) is not dict or set(value) != KEYS:
        return False
    items = value["bring"]
    return (value["place"] == reference["place"] and value["slot"] == reference["slot"]
            and type(items) is list and all(type(item) is str for item in items)
            and len(items) == len(reference["bring"]) and set(items) == set(reference["bring"]))


def score(bundle):
    expected = prompts()
    if type(bundle) is not dict or set(bundle) != set(expected):
        raise DiagnosticError("Expected exactly one observation per synthetic case")
    counts = dict.fromkeys(ARMS, 0)
    wins = losses = 0
    for case_id, case in expected.items():
        row = bundle[case_id]
        if type(row) is not dict or set(row) != set(ARMS):
            raise DiagnosticError("Both matched model arms required")
        correct = []
        for arm in ARMS:
            obs = row[arm]
            if (type(obs) is not dict or set(obs) != {"transcript_sha256", "answer"}
                    or obs["transcript_sha256"] != case["transcript_sha256"]
                    or type(obs["answer"]) is not str):
                raise DiagnosticError("Observation must attest to the same transcript")
            passed = _matches(obs["answer"], CASES[case_id][1])
            counts[arm] += passed
            correct.append(passed)
        wins += correct[1] and not correct[0]
        losses += correct[0] and not correct[1]
    return {"status": "SYNTHETIC_CHAT_STATE_DIAGNOSTIC_ONLY", "cases": len(expected),
            "correct": counts, "neral_wins": wins, "neral_losses": losses,
            "ties": len(expected) - wins - losses, "independent_holdout": False,
            "model_identity_verified": False, "release_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prompts", "score"))
    args = parser.parse_args()
    if args.mode == "prompts":
        result = prompts()
    else:
        raw = sys.stdin.buffer.read(131073)
        if len(raw) > 131072:
            parser.error("Observation bundle too large")
        try:
            result = score(_decode(raw))
        except (ValueError, RecursionError):
            parser.error("Invalid synthetic observation bundle")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
