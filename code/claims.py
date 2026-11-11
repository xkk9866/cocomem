"""Parsing of the compact claim format produced by prompts.EXTRACT."""
import re

_LINE = re.compile(r"^\s*[-*\d.<\[]*\s*([UAB])\s*[\]>]?\s*\|\s*\[?t?(\d+)\]?\s*\|\s*(.+)$")
_REV = re.compile(r"\|+\s*REVISES\s*:", re.I)        # "|| REVISES:" as the prompts ask for it, and "||REVISES:", "|REVISES:"
DEFAULT_CODES = {"U": "user", "A": "assistant"}
# the format examples of the extraction prompts (prompts.EXTRACT, prompts.EXTRACT_DIALOG) are not session content
_EXAMPLE = re.compile(r"adopted a beagle named Max on 2023/05/14|recommended the book \"Atomic Habits\" by James Clear")


def parse_claims(text, max_claims=25, codes=None):
    codes = codes or DEFAULT_CODES
    out = []
    if not text or text.strip().upper().startswith("NONE"):
        return out
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        spk, turn, body = m.group(1), int(m.group(2)), m.group(3).strip().rstrip("|>").strip()
        revises = None
        r = _REV.search(body)
        if r:
            body, revises = body[:r.start()].strip(), body[r.end():].strip()
        if len(body) < 8 or _EXAMPLE.search(body):
            continue
        if spk not in codes:
            continue
        out.append({"text": body, "speaker": codes[spk], "turn": turn, "revises": revises})
        if len(out) >= max_claims:
            break
    return out
