"""CPU checks of keyed record maintenance (commutator21.upsert_record) and of the chain metrics of suffbench5."""
import sys

from commutator21 import upsert_record, record_entries, foreign_entries, anchor_record


def eq(a, b, what):
    if a != b:
        print("FAIL", what, "\n  got ", repr(a), "\n  want", repr(b))
        sys.exit(1)
    print("ok  ", what)


R = "c1 = 43 km; c2 = 26 km; c3 = 36 km; total = 43 + 26 + 36 = 105 km"
# the operator rewrites the entry of c2 and the computation, and alters the entry of c3 by mistake
O = "c1 = 43 km; c2 = 30 km; c3 = 63 km; total = 43 + 30 + 36 = 109 km"
eq(upsert_record(R, O, "c2"), "c1 = 43 km; c2 = 30 km; c3 = 36 km; total = 43 + 30 + 36 = 109 km", "entry of another claim is carried over")
eq(foreign_entries(O, "c2") != foreign_entries(R, "c2"), True, "the altered entry is detected")
eq(upsert_record(R, R.upper().lower(), "c2"), R, "a copy keeps the record bit for bit")
eq(upsert_record(R, "c1 = 43 km; c3 = 36 km; total = 43 + 36 = 79 km", "c2"), "c1 = 43 km; c3 = 36 km; total = 43 + 36 = 79 km",
   "the corrected claim leaves the record when the operator drops it")
eq(upsert_record("c1 = 43 km; c3 = 36 km; total = 79 km", "c1 = 43 km; c2 = 5 km; c3 = 36 km; total = 84 km", "c2"),
   "c1 = 43 km; c3 = 36 km; c2 = 5 km; total = 84 km", "the corrected claim enters the record")
eq(upsert_record("c2 = Tuesday; c2 = 26 km; c4 = 50 pages; max = 50 (c4)", "c2 = Tuesday; c2 = 60 km; c4 = 5 pages; max = 60 (c2)", "c2"),
   "c2 = Tuesday; c2 = 60 km; c4 = 50 pages; max = 60 (c2)", "repeated key of the corrected claim")
eq(upsert_record("c7 = 25 postcards", "c7 = 17 postcards", "c7"), "c7 = 17 postcards", "one-operand record")
eq(upsert_record("none", "c7 = 17 postcards", "c7"), "c7 = 17 postcards", "no stored record")
eq(upsert_record(R, "none", "c2"), "none", "the operator writes no record")
eq(upsert_record("c1;c2;c3=43;26;36;total 105", "c1;c2;c3=43;30;36;total 109", "c2"), "c1;c2;c3=43;30;36;total 109",
   "a record without operand entries of other claims is the operator's")
eq(anchor_record(upsert_record(R, O, "c2"), "c2", "c9"), "c1 = 43 km; c9 = 30 km; c3 = 36 km; total = 43 + 30 + 36 = 109 km", "anchoring after the upsert")
eq([k for k, _ in record_entries("[c3] = 5; c4: 7; total = c3 + c4 = 12")], ["c3", "c4", None], "entry keys")

# the sentence stays, the entry of the corrected operand is still written (commutator21.entry_target)
from commutator21 import entry_target
C = "c1 = 43 km; c2 = 26 km; c3 = 36 km; days over 20 km = 3"
S = "The user ran more than 20 km on three days [c1][c2][c3]."
eq(entry_target(C, ["c1", "c2", "c3"], "c1 = 43 km; c2 = 30 km; c3 = 36 km; days over 20 km = 3", S, S, "c2"),
   "c1 = 43 km; c2 = 30 km; c3 = 36 km; days over 20 km = 3", "a copy of the sentence still writes the entry of the corrected claim")
eq(entry_target(C, ["c1", "c2", "c3"], "c1 = 43 km; c2 = 30 km; c3 = 63 km; days over 20 km = 4", "The user ran more than 20 km on four days.", S, "c2"),
   "c1 = 43 km; c2 = 30 km; c3 = 36 km; days over 20 km = 3",
   "when the proposal restated a sentence that stays, only the entry of the corrected claim is taken from it")
eq(entry_target(C, ["c1", "c3"], "c1 = 43 km; c2 = 30 km; c3 = 36 km", S, S, "c2"), C, "a claim that is no operand writes nothing")
eq(entry_target(C, ["c1", "c2", "c3"], "c1 = 43 km; c3 = 36 km", S, S, "c2"), C, "a proposal without the entry writes nothing")
eq(entry_target(C, ["c1", "c2", "c3"], None, None, S, "c2"), C, "no proposal writes nothing")

# error of a record along a sequence: with keyed maintenance an entry can be wrong only if the last correction of its
# own claim wrote it wrong
import random
rnd = random.Random(0)
for keyed in (True, False):
    wrong_last = []
    for trial in range(300):
        vals = {f"c{j}": rnd.randrange(3, 60) for j in range(1, 6)}
        rec = "; ".join(f"{c} = {v}" for c, v in vals.items()) + "; total = " + str(sum(vals.values()))
        last_ok = {c: True for c in vals}
        for t in range(100):
            c = rnd.choice(list(vals))
            vals[c] = rnd.randrange(3, 60)
            cur = dict(p.split(" = ") for p in rec.split("; ")[:-1])
            out = {k: (str(vals[k]) if k == c else v) for k, v in cur.items()}
            last_ok[c] = True
            if rnd.random() < 0.05:                                  # the operator writes the corrected entry wrong
                out[c], last_ok[c] = "0", False
            if rnd.random() < 0.05:                                  # the operator alters the entry of another claim
                o = rnd.choice([k for k in out if k != c])
                out[o] = "0"
            o_rec = "; ".join(f"{k} = {v}" for k, v in out.items()) + "; total = " + str(sum(vals.values()))
            rec = upsert_record(rec, o_rec, c) if keyed else o_rec
        cur = dict(p.split(" = ") for p in rec.split("; ")[:-1])
        bad = {k for k in cur if cur[k] != str(vals[k])}
        if keyed:
            eq(bad, {k for k in vals if not last_ok[k]}, f"trial {trial}") if trial < 2 else None
            assert bad == {k for k in vals if not last_ok[k]}
        wrong_last.append(len(bad) / 5)
    print("keyed" if keyed else "raw  ", "share of wrong operand entries after 100 corrections: %.3f" % (sum(wrong_last) / len(wrong_last)))

from suffbench5 import operands_ok, entry_ok
eq(entry_ok("c1 = 43 km; c2 = 30 km; total = 73", 1, "On Tuesday, the user ran 30 km."), True, "entry_ok: the entry carries the new value")
eq(entry_ok("c1 = 43 km; c2 = 26 km; total = 43 + 30 = 73", 1, "On Tuesday, the user ran 30 km."), False, "entry_ok: a stale entry")
eq(entry_ok("c1 = 43 km; total = 73", 1, "On Tuesday, the user ran 30 km."), False, "entry_ok: no entry")
eq(operands_ok("c1 = 43 km (Mon); c2 = 26 km; c3: 36 km; total = 105", [43, 26, 36]), 1.0, "operands_ok all")
eq(operands_ok("c1 = 43 km (Mon); c2 = 20 km; total = 43 + 26 + 36 = 105", [43, 26, 36]), 1 / 3, "operands_ok partial (computation entries do not count)")

# part_b of the benchmark with a simulated operator: control flow and output layout (no model)
import asyncio
import json
import re
import suffbench5
from suffbench import build

cases, units = build(6, 0)
Ds = [{k: "; ".join(f"c{j + 1} = {v} {u['unit']}" for j, v in enumerate(u["vals"])) + f"; {k} over the days" for k in suffbench5.KINDS}
      for u in units]


class FakeOps:
    sys = ["copy", "comm22", "comm22_raw", "ledger_all"]

    def __init__(self):
        self.rnd = random.Random(1)

    async def run(self, sysn, sent, rec, i, old, new, facts):
        if sysn == "copy":
            return sent, rec
        nv = re.search(r"(\d+)", new.split("user", 1)[1]).group(1)
        if sysn == "ledger_all":
            return sent + " ", rec
        ent = dict(p.split(" = ", 1) for p in rec.split("; ")[:-1])
        unit = ent[f"c{i + 1}"].split(" ", 1)[1]
        ent[f"c{i + 1}"] = f"{nv} {unit}"
        if self.rnd.random() < 0.2:                                   # the simulated operator damages another entry
            o = self.rnd.choice([k for k in ent if k != f"c{i + 1}"])
            ent[o] = f"0 {unit}"
        out = "; ".join(f"{k} = {v}" for k, v in ent.items()) + "; " + rec.split("; ")[-1]
        vals = [int(v.split()[0]) for v in ent.values()]
        s2 = re.sub(r"\d+", str(sum(vals)), sent, count=1)
        return s2, (suffbench5.upsert_record(rec, out, f"c{i + 1}") if sysn in suffbench5.KEYED else out)


out = asyncio.run(suffbench5.part_b(FakeOps(), units, Ds, 4, 30, 1))
for k in ("record_ok", "operands_ok", "step", "step_all", "step_n", "n", "length", "copy", "comm22", "comm22_raw"):
    assert k in out, k
eq(out["operands_ok"]["comm22"]["10"], 100.0, "simulated chain: keyed record keeps every operand current")
th = out["theorem"]
eq(sorted(th), ["comm22", "comm22_raw"], "theorem quantities for the systems that carry a record")
eq(th["comm22"]["1-10"]["current"], 100.0, "keyed: the record is current before every step")
eq(th["comm22"]["1-10"]["eta"], 0.0, "the simulated operator writes the corrected entry right")
assert th["comm22_raw"]["11-50"]["current"] < 50.0, th["comm22_raw"]
assert abs(th["comm22"]["1-10"]["eps"] - (100 - out["step"]["comm22"]["1-10"])) < 1e-9      # current everywhere: eps = observed
assert th["comm22"]["all"]["bound"] >= th["comm22"]["all"]["observed"] - 1e-9
assert out["operands_ok"]["comm22_raw"]["10"] < 100.0
print(json.dumps({k: out[k] for k in ("operands_ok", "step", "step_n", "copy")})[:600])
print("ALL OK")
