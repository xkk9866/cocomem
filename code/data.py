"""Dataset loading and formatting for LongMemEval-S and LoCoMo."""
import hashlib
import json
import random

ROOT = "/data/yaokang/corrmem"
LME_PATH = f"{ROOT}/data/longmemeval_s_cleaned.json"
LOCOMO_PATH = f"{ROOT}/data/locomo10.json"

DIRECT_TYPES = ("single-session-user", "single-session-assistant", "knowledge-update")
DERIVED_TYPES = ("multi-session", "temporal-reasoning")


def session_hash(turns):
    return hashlib.md5(json.dumps([[t["role"], t["content"]] for t in turns]).encode()).hexdigest()


def format_session(turns, date, max_turn_chars=None, number=True):
    lines = [f"[Session date: {date}]"]
    for i, t in enumerate(turns):
        c = t["content"].strip()
        if max_turn_chars and len(c) > max_turn_chars:
            c = c[:max_turn_chars] + " ..."
        who = {"user": "User", "assistant": "Assistant"}.get(t["role"], t["role"])
        lines.append(f"[t{i}] {who}: {c}" if number else f"{who}: {c}")
    return "\n".join(lines)


def load_lme(eligible_only=True):
    d = json.load(open(LME_PATH))
    out = []
    for x in d:
        if eligible_only and ("_abs" in x["question_id"] or x["question_type"] == "single-session-preference"):
            continue
        sessions = []
        for sid, date, turns in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]):
            sessions.append({"sid": sid, "date": date, "turns": turns, "hash": session_hash(turns),
                             "evidence": sid in x["answer_session_ids"]})
        # chronological order (dates are sortable strings 'YYYY/MM/DD (Day) HH:MM')
        order = sorted(range(len(sessions)), key=lambda i: (sessions[i]["date"][:10] + sessions[i]["date"][-5:], i))
        sessions = [sessions[i] for i in order]
        out.append({"qid": x["question_id"], "qtype": x["question_type"], "question": x["question"],
                    "answer": str(x["answer"]), "qdate": x["question_date"], "sessions": sessions})
    return out


def stratified_subset(items, n, seed=0):
    """Deterministic subset preserving the question-type proportions."""
    rng = random.Random(seed)
    by = {}
    for x in items:
        by.setdefault(x["qtype"], []).append(x)
    total = len(items)
    out = []
    for t, xs in sorted(by.items()):
        k = max(1, round(n * len(xs) / total))
        xs = sorted(xs, key=lambda z: z["qid"])
        rng.shuffle(xs)
        out += xs[:k]
    return sorted(out, key=lambda z: z["qid"])
