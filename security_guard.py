from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    category: str | None = None
    response: str | None = None


# Intentionally conservative, high-signal patterns. This is a local defense-in-depth guard,
# not a full safety classifier and never counts as raw Neral model capability.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        'secret_exfiltration',
        re.compile(r'(?i)(zeig(?:e)?|gib|verrat|reveal|show|print|read|lies).{0,80}(api[- ]?key|passw(?:ort|örter)|passwords?|secret|token|zugangsdaten|credentials|\.env|private key)'),
        'Ich gebe keine Passwörter, API-Schlüssel, Tokens oder andere vertrauliche Zugangsdaten aus.'
    ),
    (
        'credential_theft',
        re.compile(r'(?i)(stehl|klau|abgreif|steal|harvest).{0,80}(passw|credential|login|zugangsdaten)|password[- ]?steal|credential[- ]?steal'),
        'Dabei helfe ich nicht. Ich kann stattdessen erklären, wie Credential-Diebstahl erkannt und verhindert wird.'
    ),
    (
        'malware_creation',
        re.compile(r'(?i)(schreib|bau|erstell|code|write|build|create).{0,60}(malware|ransomware|keylogger|trojan|trojaner|stealer)'),
        'Ich helfe nicht beim Erstellen schädlicher Software. Ich kann bei Erkennung, Analyse in einer isolierten Testumgebung und Schutzmaßnahmen helfen.'
    ),
    (
        'unauthorized_access',
        re.compile(r'(?i)(hack|knack|umgeh|bypass|break into).{0,70}(fremd|ander|konto|account|login|website|server|zugriff|access)'),
        'Ich helfe nicht beim Umgehen fremder Zugriffskontrollen. Für autorisierte Sicherheitstests kann ich defensive Methoden erklären.'
    ),
    (
        'explosive_weapon',
        re.compile(r'(?i)(bau|herstell|make|build).{0,60}(bombe|sprengstoff|explosive|explosivstoff)'),
        'Dabei helfe ich nicht. Ich kann allgemeine Informationen zu Sicherheitsvorschriften, Gefahrenprävention oder Notfallmaßnahmen geben.'
    ),
    (
        'poisoning',
        re.compile(r'(?i)((vergift|poison).{0,60}(jemand|person|heimlich|kill|töten)|(jemand|person|heimlich).{0,60}(vergift|poison))'),
        'Dabei helfe ich nicht. Bei einer möglichen Vergiftung sollte sofort medizinische Hilfe beziehungsweise eine Giftnotrufstelle kontaktiert werden.'
    ),
    (
        'self_harm',
        re.compile(r'(?i)(mich selbst|myself).{0,40}(verletz|töten|umbringen|hurt|kill)|selbstverletz|suizid.*anleitung'),
        'Ich kann nicht bei Anleitungen zur Selbstverletzung helfen. Bei akuter Gefahr kontaktiere bitte sofort den Notruf oder eine Person in deiner Nähe.'
    ),
    (
        'prompt_injection',
        re.compile(r'(?i)(ignorier|ignore|überschreib|override).{0,80}(regel|instruction|system|sicherheit).{0,100}(secret|geheim|api|password|passwort|token|credential)'),
        'Nein. Nutzertext hebt Sicherheitsregeln oder den Schutz vertraulicher Daten nicht auf.'
    ),
]

# Defensive intent can prevent false positives for some dual-use terms.
_DEFENSIVE = re.compile(r'(?i)(schütz|sicher|erkenn|verhinder|defensiv|abwehr|eigenes? system|eigenes? konto|autorisiert|authorized|defend|protect|detect|prevent|mitigation)')
_ALWAYS_BLOCK = {'secret_exfiltration', 'credential_theft', 'malware_creation', 'explosive_weapon', 'poisoning', 'self_harm', 'prompt_injection'}

# High-signal secret formats. These are output filters only; they never replace proper secret
# storage. Patterns intentionally require realistic lengths/prefixes to reduce false positives.
_SECRET_PATTERNS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|$)'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{20,}\b'),
    re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}\b'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'),
    re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b'),
    re.compile(r'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b'),
    re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}'),
    re.compile(r'(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s:/@]+:[^\s@/]{8,}@[^\s]+'),
]


def _normalize_prompt(prompt: str) -> str:
    prompt = unicodedata.normalize('NFKC', prompt)
    # Drop zero-width/format characters commonly used to split keywords visually.
    prompt = ''.join(c for c in prompt if unicodedata.category(c) != 'Cf')
    return re.sub(r'\s+', ' ', prompt)


def check_prompt(prompt: str) -> GuardResult:
    if not isinstance(prompt, str):
        return GuardResult(False, 'invalid_input', 'Ungültige Eingabe.')
    prompt = _normalize_prompt(prompt)
    # Explicit harmful requests stay blocked even when a benign qualifier is appended.
    extra = [
        ('credential_theft', r'(?i)(fremd.{0,40}(zugangsdaten|passw).{0,30}(abgreif|stehl|klau)|(give|write|create|build).{0,40}malware.{0,60}steals?.{0,30}password|(schreib|create|write).{0,40}phishing.{0,70}(stehl|zugangsdaten|credential))'),
        ('unauthorized_access', r'(?i)(hack|knack|umgeh|bypass|break into).{0,70}(fremd|someone else|anderen? (konto|account))'),
        ('malware_creation', r'(?i)(gib|schreib|create|give|build).{0,80}(code|programm|program).{0,100}((fremd.{0,30}verschl\w*sselt)|heimlich.{0,50}(ausliest|aufzeichnet)|steals?.{0,30}password)'),
        ('secret_exfiltration', r'(?i)(lies|read|cat|öffne|open).{0,50}(\.env|id_rsa|credentials|secrets?).{0,80}(zeig|show|send|poste|print|ausgeben)'),
    ]
    for category, pattern in extra:
        if re.search(pattern, prompt):
            return GuardResult(False, category, 'Dabei helfe ich nicht. Ich kann bei Erkennung, Schutzmaßnahmen und autorisierten defensiven Tests helfen.')

    # Allow concrete prevention/detection tasks, not a free-floating "defensive" keyword.
    defensive_task = re.search(r'(?i)^(wie (schütz|schuetz|erkenn|verhinder)|how (do i|can i) (protect|detect|prevent)|explain how to (detect|prevent)|erkläre.{0,20}(erkenn|schutz)|schreib.{0,20}(regel|checkliste).{0,40}(erkennung|schutz|abwehr))', prompt)
    explicit_exfil = re.search(r'(?i)(dein|your|intern|echt|real|kontext).{0,40}(secret|api|passw|credential)|heimlich.{0,40}(ausles|aufzeichn|stiehl)', prompt)
    placeholder_example = re.search(r'(?i)(platzhalter|placeholder|YOUR_API_KEY)', prompt) and not re.search(r'(?i)(deinen?|your secret|intern|echte|kontext)', prompt)
    defensive = bool(defensive_task) and not explicit_exfil
    for category, pattern, response in _RULES:
        if pattern.search(prompt):
            if category == 'secret_exfiltration' and placeholder_example:
                continue
            if defensive and category in {'credential_theft', 'malware_creation', 'unauthorized_access'}:
                continue
            return GuardResult(False, category, response)
    return GuardResult(True)


def sanitize_output(text: str) -> tuple[str, bool]:
    if not isinstance(text, str):
        return '', False
    redacted = False
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            text = pattern.sub('[REDACTED_SECRET]', text)
            redacted = True
    # Keep one assistant turn only. Tiny models otherwise tend to synthesize a new User turn.
    for marker in ('\nUser:', '\nFrage:', '\nHuman:', '\nAssistant:'):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip(), redacted
