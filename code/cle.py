"""Counterfactual Lineage Field (CLF).

A single neural field  phi(c, u)_t = P(token t of memory unit u causally depends on claim c),
learned from the compressor LLM's own counterfactual likelihoods (cla.py):
    token label  y_t = 1[ log p(u_t | S) - log p(u_t | S \\ {c}) >= tau ]
The field is read at two resolutions with one decision rule (p >= 1/2):
    unit lineage   c in Lin(u)  <=>  max_t phi(c, u)_t >= 1/2
    repair support span(u, c)   =   sentences of u containing a token with phi >= 1/2
Input is just the pair (claim text, unit text); citation markers are stripped so the field
cannot copy the compressor's self-reported citations.

Architecture: XLM-R cross-encoder (initialised from bge-reranker-base) with a token head over
the unit segment; unit score = max-pooled token logit."""
import argparse
import json
import math
import os
import random
import re

import numpy as np
import torch
from torch import nn

BACKBONE = "/data/yaokang/corrmem/models/bge-reranker-base"
CITE = re.compile(r"\s*\[\s*(c\d+(?:\s*[,–—-]\s*c?\d+)*)\s*\]")
LEVEL_TAG = {"note": "note:", "digest": "digest:", "profile": "profile:"}


def strip_with_map(text):
    """Remove citation markers; return stripped text and old->new char index map (-1 if removed)."""
    keep = np.ones(len(text), dtype=bool)
    for m in CITE.finditer(text):
        keep[m.start():m.end()] = False
    new = "".join(ch for ch, k in zip(text, keep) if k)
    mp = np.cumsum(keep) - 1
    mp[~keep] = -1
    return new, mp


class CLFModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(path)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, input_ids, attention_mask):
        hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hs).squeeze(-1)   # token logits


def encode(tok, claims, units, max_len=512):
    """Pair encoding: segment A = claim, segment B = unit (truncated). Returns batch and unit-token masks."""
    b = tok(claims, units, truncation="only_second", max_length=max_len, padding=True,
            return_offsets_mapping=True, return_tensors="pt")
    seq = [b.sequence_ids(i) for i in range(len(claims))]
    umask = torch.tensor([[s == 1 for s in row] for row in seq])
    return b, umask


class CLF:
    def __init__(self, path, device=None, max_len=512):
        from transformers import AutoTokenizer
        self.device = device or os.environ.get("CM_CLE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = CLFModel(BACKBONE)
        self.model.load_state_dict(torch.load(os.path.join(path, "clf.pt"), map_location="cpu"))
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_len = max_len

    @torch.no_grad()
    def field(self, unit_text_cited, claim_texts, level="note", bs=32):
        """Per-claim token field over the stripped unit text: list of [(start, end, prob)] + unit scores."""
        text, _ = strip_with_map(unit_text_cited)
        a = f"{LEVEL_TAG.get(level, '')} {text}"
        off0 = len(LEVEL_TAG.get(level, "")) + 1
        fields, scores = [], []
        for i in range(0, len(claim_texts), bs):
            cs = claim_texts[i:i + bs]
            b, um = encode(self.tok, cs, [a] * len(cs), self.max_len)
            offs = b.pop("offset_mapping")
            b = {k: v.to(self.device) for k, v in b.items()}
            p = torch.sigmoid(self.model(**b).float()).cpu()
            p = p.masked_fill(~um, 0.0)
            for j in range(len(cs)):
                idx = um[j].nonzero().squeeze(-1).tolist()
                fields.append([(int(offs[j, t, 0]) - off0, int(offs[j, t, 1]) - off0, float(p[j, t])) for t in idx])
                scores.append(float(p[j].max()))
        return fields, scores

    def score(self, unit_text_cited, claim_texts, level="note"):
        return self.field(unit_text_cited, claim_texts, level)[1]


# ---------------------------------------------------------------- training
def load_rows(path, tau):
    rows = []
    for line in open(path):
        r = json.loads(line)
        text, mp = strip_with_map(r["text"])
        for c, toks in r["tokens"].items():
            spans = []
            for tk in toks:
                s, e, d = tk[0], tk[1], tk[2]
                if d >= tau and s < len(mp):
                    ns = [mp[k] for k in range(s, min(e, len(mp))) if mp[k] >= 0]
                    if ns:
                        spans.append((min(ns), max(ns) + 1))
            rows.append({"qid": r["qid"], "level": r["level"], "unit": text, "claim": r["claims"][c],
                         "spans": spans, "y": int(bool(spans)), "cited": c in r["cited"], "uid": r["uid"], "cid": c})
    return rows


def token_labels(offs, umask, spans, off0):
    y = torch.zeros(umask.shape, dtype=torch.float)
    for t in range(len(umask)):
        if not umask[t]:
            continue
        s, e = int(offs[t, 0]) - off0, int(offs[t, 1]) - off0
        if any(s < se and e > ss for ss, se in spans):
            y[t] = 1.0
    return y


def train(args):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.metrics import roc_auc_score, f1_score
    random.seed(0)
    torch.manual_seed(0)
    rows = load_rows(args.data, args.tau)
    qids = sorted({r["qid"] for r in rows})
    random.shuffle(qids)
    val_q = set(qids[:max(2, len(qids) // 8)])
    tr = [r for r in rows if r["qid"] not in val_q]
    va = [r for r in rows if r["qid"] in val_q]
    print("pairs", len(tr), len(va), "unit pos-rate", np.mean([r["y"] for r in tr]), flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = CLFModel(BACKBONE).cuda()
    model.enc.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr) / args.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()

    def batch(rs):
        units = [f"{LEVEL_TAG[r['level']]} {r['unit']}" for r in rs]
        b, um = encode(tok, [r["claim"] for r in rs], units)
        offs = b.pop("offset_mapping")
        ys = torch.stack([token_labels(offs[i], um[i], r["spans"], len(LEVEL_TAG[r["level"]]) + 1)
                          for i, r in enumerate(rs)])
        yu = torch.tensor([float(ys[i].max() > 0) for i in range(len(rs))])
        return {k: v.cuda() for k, v in b.items()}, um.cuda(), ys.cuda(), yu.cuda()

    def evaluate():
        model.eval()
        P, Y, TP, TY = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(va), 32):
                b, um, ys, yu = batch(va[i:i + 32])
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = model(**b).float()
                lg = lg.masked_fill(~um, -1e4)
                P += torch.sigmoid(lg.max(-1).values).tolist()
                Y += yu.tolist()
                TP += torch.sigmoid(lg[um]).tolist()
                TY += ys[um].tolist()
        model.train()
        return (roc_auc_score(Y, P), f1_score(Y, [p >= 0.5 for p in P]),
                roc_auc_score(TY, TP), f1_score(TY, [p >= 0.5 for p in TP]))

    pos_t = max(1e-3, float(np.mean([len(r["spans"]) > 0 for r in tr])) * 0.1)
    pw_t = torch.tensor(min(10.0, (1 - pos_t) / pos_t)).cuda()
    model.train()
    for ep in range(args.epochs):
        random.shuffle(tr)
        for i in range(0, len(tr), args.bs):
            b, um, ys, yu = batch(tr[i:i + args.bs])
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(**b)
            lg = lg.float()
            lt = nn.functional.binary_cross_entropy_with_logits(lg[um], ys[um], pos_weight=pw_t)
            lu = nn.functional.binary_cross_entropy_with_logits(lg.masked_fill(~um, -1e4).max(-1).values, yu)
            loss = lt + lu
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            if (i // args.bs) % 100 == 0:
                print(f"ep {ep} step {i // args.bs} loss {loss.item():.4f} (tok {lt.item():.4f} unit {lu.item():.4f})", flush=True)
        m = evaluate()
        print(f"epoch {ep} val unit-AUC {m[0]:.4f} unit-F1 {m[1]:.4f} token-AUC {m[2]:.4f} token-F1 {m[3]:.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, "clf.pt"))
    tok.save_pretrained(args.out)
    json.dump({"tau": args.tau, "val": m, "n_train": len(tr), "n_val": len(va), "val_qids": sorted(val_q)},
              open(os.path.join(args.out, "clf_meta.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/yaokang/corrmem/data/cla_dev.jsonl")
    ap.add_argument("--out", default="/data/yaokang/corrmem/models/clf")
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    train(ap.parse_args())
