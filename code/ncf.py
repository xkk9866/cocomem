"""Neural Commutator Field (NCF).

For a memory unit u (a sequence of sentences s_1..s_K) and a claim c, the commutator of
compression C and the correction operator delta_c is supported on the sentences whose content
would differ had c been stated differently (cfgen.py).  NCF is one neural field

    Phi(c, u)_k = P( s_k  in  supp [C, delta_c](u) )

read at two resolutions with one decision rule (Phi >= 1/2):
    lineage       c in Lin(u)      <=>  max_k Phi(c, u)_k >= 1/2      (which units to touch)
    repair span   Span(u, c)        =   {k : Phi(c, u)_k >= 1/2}       (which sentences to regenerate)

Input is only the pair (claim, unit) exactly as they are stored: the claim with its ledger
address "[c_i] text" and the unit text with the compressor's own citation markers.  Declared
provenance is thus *evidence*, not ground truth: the field learns to keep true citations, reject
spurious ones and add the derived dependencies that the compressor never cites (totals, temporal
spans, profile generalizations).  (Ablation ncf_v1: markers stripped, claim without address.)  Architecture: XLM-R cross-encoder initialised from
bge-reranker-base; a token head is max-pooled within each sentence."""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from torch import nn

from cocomem import split_sentences


MASK = "?"


def claim_input(cid, text, new=None):
    """The correction query: the claim at its ledger address and what it is corrected to.
    new=None asks the *marginal* question (the corrected value is unknown, written as "?")."""
    return f"[{cid}] {text} => {new if new is not None else MASK}"

BACKBONE = "/data/yaokang/corrmem/models/bge-reranker-base"
TAG = {"note": "note:", "digest": "digest:", "profile": "profile:"}


class NCFModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(path)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.hidden_size, 1))

    def forward(self, input_ids, attention_mask):
        return self.head(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state).squeeze(-1)


def encode(tok, claims, units, levels, ranges, max_len=512):
    """Returns the batch and a [B, T, Kmax] token->sentence membership tensor (unit segment only)."""
    texts = [f"{TAG[l]} {u}" for u, l in zip(units, levels)]
    b = tok(claims, texts, truncation="longest_first", max_length=max_len, padding=True,
            return_offsets_mapping=True, return_tensors="pt")
    offs = b.pop("offset_mapping")
    K = max(len(r) for r in ranges)
    M = torch.zeros(len(claims), offs.shape[1], max(K, 1), dtype=torch.bool)
    for i in range(len(claims)):
        seq = b.sequence_ids(i)
        o = len(TAG[levels[i]]) + 1
        for t, sid in enumerate(seq):
            if sid != 1:
                continue
            s, e = int(offs[i, t, 0]) - o, int(offs[i, t, 1]) - o
            for k, (a, z) in enumerate(ranges[i]):
                if s < z and e > a:
                    M[i, t, k] = True
                    break
    return b, M


def sentence_logits(tok_logits, M):
    """Max-pool token logits within each sentence; sentences with no visible token get -inf."""
    x = tok_logits.unsqueeze(-1).masked_fill(~M, -1e4)
    return x.max(1).values   # [B, K]


class NCF:
    def __init__(self, path, device=None, max_len=512):
        from transformers import AutoTokenizer
        self.device = device or os.environ.get("CM_CLE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = NCFModel(BACKBONE)
        self.model.load_state_dict(torch.load(os.path.join(path, "ncf.pt"), map_location="cpu"))
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_len = max_len

    @torch.no_grad()
    def field(self, unit_text_cited, claim_texts, level="note", bs=32):
        """[n_claims, n_sentences] probabilities over the sentences of split_sentences(unit_text_cited)."""
        text, rng = unit_text_cited, split_sentences(unit_text_cited)
        out = []
        for i in range(0, len(claim_texts), bs):
            cs = claim_texts[i:i + bs]
            b, M = encode(self.tok, cs, [text] * len(cs), [level] * len(cs), [rng] * len(cs), self.max_len)
            b = {k: v.to(self.device) for k, v in b.items()}
            lg = sentence_logits(self.model(**b).float(), M.to(self.device))
            out.append(torch.sigmoid(lg).cpu().numpy()[:, :len(rng)])
        return np.concatenate(out, 0) if out else np.zeros((0, len(rng)))


# ------------------------------------------------------------------------------------------ training
def load_rows(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        rng = split_sentences(r["text"])
        if len(rng) != len(r["y"]):
            continue
        rows.append({"qid": r["qid"], "level": r["level"], "unit": r["text"], "ranges": rng,
                     "claim": claim_input(r["cid"], r["claim"], r["cf_claim"]),
                     "claim_m": claim_input(r["cid"], r["claim"]), "old": r["claim"], "y": r["y"], "cited": r["cited"],
                     "uid": r["uid"], "cid": r["cid"]})
    return rows


def batches(tok, rows, bs, shuffle, mode="cond"):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        rs = [rows[j] for j in idx[i:i + bs]]
        if mode == "dropout":     # correction dropout: the corrected value is hidden for half the queries
            qs = [r["claim"] if random.random() < 0.5 else r["claim_m"] for r in rs]
        else:
            qs = [r["claim"] if mode == "cond" else r["claim_m"] for r in rs]
        b, M = encode(tok, qs, [r["unit"] for r in rs], [r["level"] for r in rs],
                      [r["ranges"] for r in rs])
        K = M.shape[2]
        Y = torch.zeros(len(rs), K)
        V = torch.zeros(len(rs), K, dtype=torch.bool)
        for j, r in enumerate(rs):
            for k, y in enumerate(r["y"][:K]):
                Y[j, k] = y
                V[j, k] = bool(M[j, :, k].any())   # sentence visible after truncation
        yield rs, b, M, Y, V


@torch.no_grad()
def predict(model, tok, rows, bs=48, mode="cond"):
    model.eval()
    P = []
    for rs, b, M, Y, V in batches(tok, rows, bs, False, mode):
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(**b)
        p = torch.sigmoid(sentence_logits(lg.float(), M.cuda())).cpu()
        for j, r in enumerate(rs):
            P.append([float(p[j, k]) if V[j, k] else 0.0 for k in range(len(r["y"]))])
    model.train()
    return P


def metrics(rows, P):
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    ys = [y for r in rows for y in r["y"]]
    ps = [p for pr in P for p in pr]
    yu = [int(any(r["y"])) for r in rows]
    pu = [max(pr) if pr else 0.0 for pr in P]
    out = {}
    for name, Y, Q in (("sent", ys, ps), ("unit", yu, pu)):
        pr, rc, f1, _ = precision_recall_fscore_support(Y, [q >= 0.5 for q in Q], average="binary", zero_division=0)
        out[name] = {"auc": roc_auc_score(Y, Q) if len(set(Y)) > 1 else float("nan"), "p": pr, "r": rc, "f1": f1,
                     "pos": float(np.mean(Y)), "n": len(Y)}
    return out


def train(a):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = load_rows(a.data)
    qids = sorted({r["qid"] for r in rows})
    random.shuffle(qids)
    val_q = set(qids[:max(2, len(qids) // 8)])
    tr = [r for r in rows if r["qid"] not in val_q]
    va = [r for r in rows if r["qid"] in val_q]
    pos = float(np.mean([y for r in tr for y in r["y"]]))
    print(f"pairs train {len(tr)} val {len(va)}  sentence pos-rate {pos:.3f}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = NCFModel(BACKBONE).cuda()
    model.enc.gradient_checkpointing_enable()
    model.enc.embeddings.word_embeddings.weight.requires_grad_(False)   # 192M-row vocabulary table stays frozen
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    pw = torch.tensor((1 - pos) / pos).cuda()   # class balance from the data, not a tuned constant
    best, hist = -1, []
    for ep in range(a.epochs):
        for step, (rs, b, M, Y, V) in enumerate(batches(tok, tr, a.bs, True, "dropout")):
            b = {k: v.cuda() for k, v in b.items()}
            M, Y, V = M.cuda(), Y.cuda(), V.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(**b)
            sl = sentence_logits(lg.float(), M)
            ls = nn.functional.binary_cross_entropy_with_logits(sl[V], Y[V], pos_weight=pw)
            ul = sl.masked_fill(~V, -1e4).max(1).values
            lu = nn.functional.binary_cross_entropy_with_logits(ul, (Y * V).max(1).values)
            loss = ls + lu
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            if step % 100 == 0:
                print(f"ep {ep} step {step} loss {loss.item():.4f}", flush=True)
        m = metrics(va, predict(model, tok, va, mode="cond"))
        m["marginal"] = metrics(va, predict(model, tok, va, mode="marg"))
        hist.append(m)
        print(f"epoch {ep} val {json.dumps(m)}", flush=True)
        if m["sent"]["auc"] > best:
            best = m["sent"]["auc"]
            os.makedirs(a.out, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(a.out, "ncf.pt"))
            tok.save_pretrained(a.out)
    json.dump({"val": hist, "n_train": len(tr), "n_val": len(va), "val_qids": sorted(val_q), "pos_rate": pos},
              open(os.path.join(a.out, "ncf_meta.json"), "w"), indent=1)


def evaluate(a):
    """Lineage quality on held-out users against counterfactual ground truth, vs. alternatives."""
    from transformers import AutoTokenizer
    from embed import Embedder
    rows = load_rows(a.data)
    tok = AutoTokenizer.from_pretrained(a.out)
    model = NCFModel(BACKBONE).cuda()
    model.load_state_dict(torch.load(os.path.join(a.out, "ncf.pt"), map_location="cpu"))
    Pn = predict(model, tok, rows, mode="cond")
    Pm = predict(model, tok, rows, mode="marg")
    res = {"ncf": metrics(rows, Pn), "ncf_marginal": metrics(rows, Pm)}
    # self-reported provenance: the compressor's own citation markers
    res["self_cite"] = metrics(rows, [[float(c) for c in r["cited"]] for r in rows])
    # construction lineage: every input claim reaches every sentence
    res["coarse"] = metrics(rows, [[1.0] * len(r["y"]) for r in rows])
    # semantic similarity between claim and sentence (embedding retrieval, Mem0/A-Mem style)
    E = Embedder.get()
    cl = E.encode([r["old"] for r in rows], True)
    P = []
    for i, r in enumerate(rows):
        ss = E.encode([r["unit"][x:y] for x, y in r["ranges"]], False)
        P.append([float((s @ cl[i] + 1) / 2) for s in ss])
    res["similarity"] = metrics(rows, P)
    by = {}
    for lev in ("note", "digest", "profile"):
        idx = [i for i, r in enumerate(rows) if r["level"] == lev]
        if idx:
            by[lev] = {"ncf": metrics([rows[i] for i in idx], [Pn[i] for i in idx]),
                       "ncf_marginal": metrics([rows[i] for i in idx], [Pm[i] for i in idx]),
                       "self_cite": metrics([rows[i] for i in idx], [[float(c) for c in rows[i]["cited"]] for i in idx])}
    res["by_level"] = by
    res["thr_sweep"] = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        res["thr_sweep"][str(t)] = metrics(rows, [[float(p >= t) * 0.5 + 0.25 * (p >= t) for p in pr] for pr in Pn])
    with open(a.report.replace(".json", "_pred.jsonl"), "w") as f:
        for r, pr in zip(rows, Pn):
            f.write(json.dumps({"qid": r["qid"], "uid": r["uid"], "cid": r["cid"], "level": r["level"], "y": r["y"],
                                "ncf": pr, "cited": r["cited"]}) + "\n")
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--data", default="/data/yaokang/corrmem/data/cf_dev.jsonl")
    ap.add_argument("--out", default="/data/yaokang/corrmem/models/ncf")
    ap.add_argument("--report", default="/data/yaokang/corrmem/results/lineage_quality.json")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
