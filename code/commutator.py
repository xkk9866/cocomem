"""Neural Commutator T_theta: distillation of the compressor's response to a correction.

A memory unit u = C(I) is a sequence of sentences produced by the compressor C (Qwen3-8B) from its
input claims I.  A correction delta = (c -> c') changes one input.  The compressor's own answer to
"what would sentence k have said under the corrected input" is one Jacobi step of greedy decoding,
teacher-forced on the stored prefix (cfgen.py):
        J_k(u; delta) = f_k(I[c -> c'] ; u_<k)         (= u_k  when sentence k does not depend on c)
T_theta learns this map with exactly two inputs, the correction and the stored sentence:
        T_theta(delta, u_k)  ~  J_k(u; delta)
Detection (does u_k depend on c?) and revision (what does it become?) are one conditional
generation: copying u_k is the "independent" outcome.  Because delta carries the new value c',
the target is determined by the inputs (up to the compressor's own decoding), unlike a
correction-agnostic lineage score Phi(c, u).  The per-call input size is |delta| + |u_k|, which does
not grow with the history.  Supervision comes only from the compressor (no human labels).

Sub-commands
  data   build train / test jsonl from cf_*.jsonl
  train  LoRA-distil the student (Qwen3-4B), then merge (served int4 by vLLM, see start_tnet.sh)
  score  copy log-likelihood score on the held-out set (AUC of dependence detection)
  eval   generate with a served model (student or zero-shot teacher) and score detection + revision
"""
import argparse
import asyncio
import json
import math
import os
import random
import re

ROOT = __import__("os").environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
BASE = "/home/yaokang/sra/models/Qwen3-4B"     # student (bf16 for training; served int4)
OUT = f"{ROOT}/models/tnet"

T_PROMPT = """A memory sentence was written from facts that included the fact below, which has now been corrected.
Correction: {old} => {new}
Sentence: {sent}
Write the sentence as it should read after the correction. If the sentence does not depend on the corrected fact, copy it unchanged.
Output only the sentence itself, keeping its citations; no explanation."""


def norm(s):
    s = re.sub(r"\[c\d+(?:,\s*c\d+)*\]", "", s)
    # Keep numeric punctuation: 5 -> -5 and 1.5 -> 15 are real edits.
    # Tokenizing also keeps sentence-final punctuation from creating edits.
    s = s.lower().replace("\u2212", "-")
    return " ".join(re.findall(r"[+-]?\d+(?:[.,:/-]\d+)*(?:%|\b)|[a-z]+", s))


# ---------------------------------------------------------------- v20: witness-conditioned commutator
# T_theta(delta, s, W): the correction, the stored sentence and the CURRENT texts of the sentence's derivation
# witness (the other claims it was derived from, dereferenced in the ledger at repair time).  With a sufficient
# witness the sentence is self-maintainable (Theorem 1), so one network both decides whether the sentence changes
# and computes what it becomes, also for totals, extremes, orders and conclusions over several claims.
W_PROMPT = """A memory sentence was derived from facts that included the fact below, which has now been corrected.
Correction: {old} => {new}
Other facts the sentence was derived from (current values):
{facts}
Sentence: {sent}
Write the sentence as it should read after the correction. Update every value, total, count, maximum or minimum, order, date, duration or conclusion that depends on the corrected fact, using the other facts where they are needed. If the sentence does not depend on the corrected fact, copy it unchanged.
Output only the sentence itself, keeping its citations; no explanation."""

WFACT_CHARS = 9000          # the student's context is 4096 tokens; a witness longer than this is cut (reported)


def render_wfacts(lines):
    """lines: '[cid] (date, speaker) text' of the witness claims other than the corrected one."""
    out, n = [], 0
    for l in lines:
        if n + len(l) > WFACT_CHARS:
            break
        out.append(l)
        n += len(l) + 1
    return "\n".join(out) if out else "(none)"


def wprompt(e):
    return W_PROMPT.format(old=e["old"], new=e["new"], facts=e.get("wfacts") or "(none)", sent=e["sent"])


def prompt_text(e):
    """The student's input: stored with the example when the data were built for a specific variant."""
    return e["prompt"] if e.get("prompt") else PROMPT(e)


def split(text):
    from cocomem import split_sentences
    return [text[a:b].strip() for a, b in split_sentences(text)]


def delta_text(cid, old):
    return f"[{cid}] {old}"


def rows_to_examples(rows):
    """One example per (correction, sentence): target = compressor's counterfactual sentence if it
    changed (y=1), else the stored sentence (copy)."""
    ex = []
    for r in rows:
        sents = split(r["text"])
        for k, s in enumerate(sents):
            if k >= len(r["y"]) or not s:
                continue
            y = int(r["y"][k])
            tgt = r["cf_sents"][k].strip() if y else s
            if y and not tgt:
                continue
            ex.append({"qid": r["qid"], "uid": r["uid"], "level": r["level"], "k": k, "y": y,
                       "old": delta_text(r["cid"], r["claim"]), "new": r["cf_claim"], "sent": s, "target": tgt,
                       "cited": int(r["cited"][k]) if k < len(r["cited"]) else 0})
    return ex


def prompt_of(e):
    return T_PROMPT.format(old=e["old"], new=e["new"], sent=e["sent"])


def cmd_data(a):
    tr, te = [], []
    for f in a.train.split(","):
        tr += [json.loads(l) for l in open(f)]
    te = [json.loads(l) for l in open(a.test)]
    tq = {r["qid"] for r in te}
    tr = [r for r in tr if r["qid"] not in tq]          # no user overlap with the held-out set
    E_tr, E_te = rows_to_examples(tr), rows_to_examples(te)
    os.makedirs(OUT, exist_ok=True)
    for name, E in (("train", E_tr), ("test", E_te)):
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for e in E:
                f.write(json.dumps(e) + "\n")
        print(name, len(E), "pos", sum(e["y"] for e in E), "users", len({e["qid"] for e in E}))


def chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def merge_adapter_cpu():
    """The saved adapter merged into the bf16 base, on the CPU.  The base is loaded with no device map: a model that
    carries one is re-dispatched by PEFT over whatever the shared GPU has free, the rest is offloaded, and an
    offloaded model cannot be saved."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    return PeftModel.from_pretrained(base, f"{OUT}/adapter", torch_device="cpu").merge_and_unload()


def cmd_train(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    import time as _time
    t_start = _time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    qlora = bool(getattr(a, "qlora", False))
    if getattr(a, "merge_only", False):
        # the adapter of a finished run is on disk and only the merge is missing (the merge was interrupted)
        stats = json.load(open(f"{OUT}/adapter/train_stats.json"))
        model = merge_adapter_cpu()
        model.config.use_cache = True
        model.save_pretrained(f"{OUT}/merged", safe_serialization=True)
        tok.save_pretrained(f"{OUT}/merged")
        json.dump(vars(a) | stats | {"qlora": True}, open(f"{OUT}/train_meta.json", "w"), indent=1)
        print("saved (merge only)", flush=True)
        return
    if qlora:
        # QLoRA: the frozen base is held in 4-bit NF4 so that the same LoRA recipe (r, alpha, lr, batch, epochs)
        # trains on a shared GPU with a few GB free; the adapter is merged into the bf16 base afterwards
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, torch_dtype=torch.bfloat16,
                                                     device_map={"": 0})
        model.config.use_cache = False
        for p_ in model.parameters():            # bf16 norms/embeddings as in the bf16 recipe (no fp32 upcast)
            p_.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
        model.config.use_cache = False       # no KV cache during checkpointed training
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    E = [json.loads(l) for l in open(f"{OUT}/train.jsonl")]
    random.Random(a.seed).shuffle(E)
    seqs, skipped = [], 0
    for e in E:
        p = tok(chat(tok, prompt_text(e)), add_special_tokens=False)["input_ids"]
        t = tok(e["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p) + len(t) > a.max_len:          # never truncate a target; over-long examples are dropped (counted)
            skipped += 1
            continue
        seqs.append((p + t, [-100] * len(p) + t))
    if not seqs:
        raise ValueError(f"No training examples fit max_len={a.max_len}; skipped {skipped}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    mb = a.bs // a.accum                      # micro-batch; the optimizer step still sees a.bs sequences
    steps = math.ceil(len(seqs) / a.bs) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    model.train()
    step = 0
    for ep in range(a.epochs):
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))     # length-bucketed batches
        batches = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        random.Random(a.seed + ep).shuffle(batches)
        for bb in batches:
            target_tokens = sum(sum(y != -100 for y in seqs[i][1]) for i in bb)
            micro, cur = [], []                  # micro-batches bounded by padded tokens (long witness prompts)
            for i in bb:
                if cur and (len(cur) >= mb or max(len(seqs[j][0]) for j in cur + [i]) * (len(cur) + 1) > a.tok_budget):
                    micro.append(cur)
                    cur = []
                cur.append(i)
            micro.append(cur)
            for b in micro:
                L = max(len(seqs[i][0]) for i in b)
                ids = torch.full((len(b), L), tok.pad_token_id)
                lab = torch.full((len(b), L), -100)
                att = torch.zeros((len(b), L), dtype=torch.long)
                for j, i in enumerate(b):
                    x, y = seqs[i]
                    ids[j, :len(x)], lab[j, :len(y)], att[j, :len(x)] = torch.tensor(x), torch.tensor(y), 1
                loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
                # HF loss is a target-token mean. Weight by target tokens,
                # not sequence count, so adaptive microbatches preserve it.
                (loss * (lab != -100).sum().item() / target_tokens).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)
    if qlora:
        # merge the adapter into the bf16 base on the CPU (the quantized copy on the GPU is discarded first)
        model.save_pretrained(f"{OUT}/adapter")
        json.dump({"n_train": len(seqs), "steps": steps, "skipped_long": skipped, "train_sec": _time.time() - t_start},
                  open(f"{OUT}/adapter/train_stats.json", "w"))
        del model, opt
        torch.cuda.empty_cache()
        model = merge_adapter_cpu()
    else:
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(f"{OUT}/merged", safe_serialization=True)
    tok.save_pretrained(f"{OUT}/merged")
    json.dump(vars(a) | {"n_train": len(seqs), "steps": steps, "skipped_long": skipped, "qlora": qlora,
                         "train_sec": _time.time() - t_start}, open(f"{OUT}/train_meta.json", "w"), indent=1)
    print("saved", flush=True)


def auc(scores, ys):
    pos = [s for s, y in zip(scores, ys) if y]
    neg = [s for s, y in zip(scores, ys) if not y]
    if not pos or not neg:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(pos + neg)     # average ranks for ties
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cmd_score(a):
    """Dependence score = -(mean log-likelihood of copying the sentence).  Threshold-free AUC."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = a.model or f"{OUT}/merged"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    E = [json.loads(l) for l in open(f"{OUT}/test.jsonl")]
    sc = []
    with torch.no_grad():
        for i in range(0, len(E), 4):
            b = E[i:i + 4]
            xs, ms = [], []
            for e in b:
                p = tok(chat(tok, prompt_text(e)), add_special_tokens=False)["input_ids"]
                t = tok(e["sent"] + tok.eos_token, add_special_tokens=False)["input_ids"]
                xs.append(p + t); ms.append([0] * len(p) + [1] * len(t))
            L = max(len(x) for x in xs)
            ids = torch.full((len(b), L), tok.pad_token_id); att = torch.zeros((len(b), L), dtype=torch.long)
            msk = torch.zeros((len(b), L))
            for j, (x, m) in enumerate(zip(xs, ms)):
                ids[j, :len(x)], att[j, :len(x)], msk[j, :len(m)] = torch.tensor(x), 1, torch.tensor(m, dtype=torch.float)
            lg = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, :-1]
            tl = lg.gather(-1, ids[:, 1:].cuda().unsqueeze(-1)).squeeze(-1).float() - torch.logsumexp(lg.float(), -1)
            m = msk[:, 1:].cuda()
            sc += (-(tl * m).sum(1) / m.sum(1)).tolist()
    ys = [e["y"] for e in E]
    res = {"auc": auc(sc, ys), "n": len(E), "pos": sum(ys)}
    for lev in ("note", "digest", "profile"):
        idx = [i for i, e in enumerate(E) if e["level"] == lev]
        res[lev] = auc([sc[i] for i in idx], [ys[i] for i in idx])
    print(json.dumps(res), flush=True)
    json.dump({"res": res, "scores": sc}, open(f"{ROOT}/results/tnet_score{a.suffix}.json", "w"))


RELEVANT = """A fact in a memory was corrected.
Before: {old}
After: {new}

Two versions of one memory sentence:
A: {a}
B: {b}

Do A and B differ in what they say about this fact, or about anything that follows from it? Differences about other topics do not count. Answer yes or no only."""


def first_line(t):
    return t.strip().split("\n")[0].strip().strip('"')


def out_budget(sent):
    return min(int(len(sent) / 3.2) + 64, 1000)


REWRITE = """A: {a}
B: {b}

Is B a version of memory sentence A, that is, a statement of the same kind about the same subject (possibly with different details), rather than a comment or explanation about A? Answer yes or no only."""


async def _relabel(a):
    """Information-level targets (Definition 1).  The compressor, used as the same two-input local
    operator, proposes the corrected sentence; the proposal is the target only if it changes what the
    sentence says about the corrected fact or its consequences (otherwise: copy).  This removes the
    decoding drift that confounds sentence-level re-derivation targets (jacobi2.py)."""
    from llm import LLM, gather_limited
    from cocomem import strip_cites
    llm = LLM("qwen3-8b", 18431, concurrency=192)
    os.makedirs(OUT, exist_ok=True)
    for name in ("train", "test"):
        E = [json.loads(l) for l in open(f"{a.src}/{name}.jsonl")]

        async def one(e):
            out = first_line(await llm.chat(prompt_of(e), max_tokens=out_budget(e["sent"]) + (3000 if a.think else 0), think=a.think))
            e = dict(e, y_j=e["y"], target_j=e["target"])
            if not out or norm(out) == norm(e["sent"]):
                return dict(e, y=0, target=e["sent"])
            old = re.sub(r"^\[c\d+\]\s*", "", e["old"])
            w = await llm.chat(REWRITE.format(a=strip_cites(e["sent"]), b=strip_cites(out)), max_tokens=4)
            if "yes" not in w.lower():           # a comment about the sentence is not a corrected sentence
                return dict(e, y=0, target=e["sent"], rejected=out)
            r = await llm.chat(RELEVANT.format(old=old, new=e["new"], a=strip_cites(e["sent"]), b=strip_cites(out)),
                               max_tokens=4)
            return dict(e, y=1, target=out) if "yes" in r.lower() else dict(e, y=0, target=e["sent"])

        R = await gather_limited([one(e) for e in E], desc=f"relabel-{name}", every=5000)
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for e in R:
                f.write(json.dumps(e) + "\n")
        print(name, len(R), "pos", sum(e["y"] for e in R), "agree_with_J",
              sum(e["y"] == e["y_j"] for e in R) / len(R), flush=True)


LEDGER = "[LEDGER]"
T_LEDGER = """A memory sentence was written from the facts below, one of which has now been corrected.

Facts (after the correction):
{facts}

Correction: {old} => {new}
Sentence: {sent}
Write the sentence as it should read after the correction, using the facts for anything that follows from the corrected fact (totals, counts, maxima, orders, dates, conclusions). If the sentence does not depend on the corrected fact, copy it unchanged."""


def unit_facts(m, u, cid, new):
    """The unit's anchored input claims after the correction (cid replaced by its corrected text)."""
    out = []
    for c in u["inputs"]:
        if c in m.claims:
            x = m.claims[c]
            out.append(f"[{c}] ({x['date'][:10]}, {x['speaker']}) {new if c == cid else x['text']}")
    return "\n".join(out)
T_PROMPT3 = T_PROMPT + "\nIf the sentence alone does not determine how it should read after the correction, answer " + LEDGER + "."
SAME_ABOUT = """A fact in a memory was corrected.
Before: {old}
After: {new}

Two versions of one memory sentence after the correction:
A: {a}
B: {b}

Do A and B state the same about this fact and about everything that follows from it? Answer yes or no only."""


def prompt3(e):
    return T_PROMPT3.format(old=e["old"], new=e["new"], sent=e["sent"])


async def _relabel3(a):
    """Sufficiency labels.  For every counterfactual correction of a unit, the compressor repairs ALL its sentences
    reading the unit's anchored claims (the ledger repair, ledger_call).  A sentence is locally sufficient iff the
    local, information-level target (from relabel) states the same about the corrected fact as the ledger repair;
    otherwise its target is LEDGER.  The ledger repair also checks the correctness of every local target."""
    from cocomem import CoCoMem, ledger_call, strip_cites
    from llm import LLM, gather_limited
    llm = LLM("qwen3-8b", 18431, concurrency=160)
    mems = {}
    os.makedirs(OUT, exist_ok=True)
    for name in ("train", "test"):
        E = [json.loads(l) for l in open(f"{a.src}/{name}.jsonl")]
        groups = {}
        for i, e in enumerate(E):
            groups.setdefault((e["qid"], e["uid"], e["old"], e["new"]), []).append(i)

        async def one(key, idx):
            qid, uid, old, new = key
            if qid not in mems:
                mems[qid] = CoCoMem.from_json(llm, json.load(open(f"{ROOT}/states/cocomem/{qid}.json")))
            m = mems[qid]
            u = m.units.get(uid)
            if u is None:
                return
            cid = re.match(r"\[(c\d+)\]", old).group(1)
            lines = []
            for c in u["inputs"]:
                if c in m.claims:
                    x = m.claims[c]
                    lines.append(f"[{c}] ({x['date'][:10]}, {x['speaker']}) {new if c == cid else x['text']}")
            K = max(E[i]["k"] for i in idx) + 1
            sents = [""] * K
            for i in idx:
                sents[E[i]["k"]] = E[i]["sent"]
            ids = [E[i]["k"] for i in idx]
            got = await ledger_call(llm, u["level"], old, new, "\n".join(lines), sents, ids)
            plain_old = re.sub(r"^\[c\d+\]\s*", "", old)
            for i in idx:
                e = E[i]
                full = got.get(e["k"], e["sent"])
                e["t_ledger"] = full
                if norm(full) == norm(e["target"]):
                    continue
                r = await llm.chat(SAME_ABOUT.format(old=plain_old, new=new, a=strip_cites(e["target"]),
                                                     b=strip_cites(full)), max_tokens=4)
                if "yes" not in r.lower():
                    E[i] = dict(e, y=2, target=LEDGER)

        await gather_limited([one(k_, v) for k_, v in groups.items()], desc=f"relabel3-{name}", every=500)
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for e in E:
                f.write(json.dumps(e) + "\n")
        print(name, len(E), "copy", sum(e["y"] == 0 for e in E), "local", sum(e["y"] == 1 for e in E),
              "ledger", sum(e["y"] == 2 for e in E), flush=True)


VERIFY = """Facts of a memory unit, after one of them was corrected:
{facts}

Correction: {old} => {new}
Memory sentence written before the correction: {sent}
Candidate for the sentence after the correction: {cand}

Is the candidate exactly what the sentence should say after the correction: everything it states agrees with the facts (including totals, counts, extremes, orders and conclusions that follow from them), no value replaced by the correction remains, and nothing else the sentence said is lost? Answer yes or no."""


def edit_ratio(a, b):
    import difflib
    return 1.0 - difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


async def _relabel_cv(a):
    """Commutation-verified targets (v10).  The commutation square has two paths: the local one T(delta, s) and
    the global one C(I[c -> c'])_k.  For every (delta, s) we gather the aligned candidate rewrites produced by the
    earlier teachers -- copy, information-level (tnet5), reasoning (tnet7) and fact-reading (ledger, tnet3's
    t_ledger) -- and let the compressor, reading the PRIVILEGED corrected inputs I[c -> c'] that the student never
    sees, score how well each candidate closes the square: v(y) = P(yes | I', delta, s, y).  The target is
    argmax v; among equal scores the minimal change to s wins (AGM minimal revision).  The student keeps exactly
    two inputs (delta, s): the privileged information shapes only its supervision."""
    from openai import AsyncOpenAI
    from cocomem import CoCoMem, strip_cites
    from llm import LLM, gather_limited
    llm = LLM("qwen3-8b", 18431, concurrency=160)
    cli = AsyncOpenAI(base_url="http://localhost:18431/v1", api_key="EMPTY", timeout=1800)
    sem = asyncio.Semaphore(192)
    mems = {}
    src = {k: f"{ROOT}/models/{k}" for k in ("tnet3", "tnet5", "tnet7")}
    os.makedirs(OUT, exist_ok=True)

    async def p_yes(prompt):
        async with sem:
            for attempt in range(6):
                try:
                    r = await cli.chat.completions.create(
                        model="qwen3-8b", messages=[{"role": "user", "content": prompt}], max_tokens=1,
                        temperature=0.0, logprobs=True, top_logprobs=10,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    await asyncio.sleep(5 * (attempt + 1))
        py = pn = 0.0
        for t in r.choices[0].logprobs.content[0].top_logprobs:
            w = t.token.strip().lower()
            if w == "yes":
                py += math.exp(t.logprob)
            elif w == "no":
                pn += math.exp(t.logprob)
        return py / (py + pn) if py + pn > 0 else 0.5

    for name in ("train", "test"):
        D = {k: [json.loads(l) for l in open(f"{v}/{name}.jsonl")] for k, v in src.items()}
        L, I5, I7 = D["tnet3"], D["tnet5"], D["tnet7"]

        async def one(i):
            e3, e5, e7 = L[i], I5[i], I7[i]
            assert (e3["qid"], e3["uid"], e3["k"], e3["old"]) == (e5["qid"], e5["uid"], e5["k"], e5["old"])
            s = e5["sent"]
            cands = [s, e5["target"], e7["target"], e3.get("t_ledger", s)]
            uniq = []
            for c in cands:
                if c and c != LEDGER and norm(c) and norm(c) not in {norm(x) for x in uniq}:
                    uniq.append(c)
            base = dict(e5, y5=e5["y"], target5=e5["target"], target7=e7["target"], t_ledger=e3.get("t_ledger"))
            if len(uniq) <= 1:
                return dict(base, y=0, target=s, scores=[1.0])
            if e5["qid"] not in mems:
                mems[e5["qid"]] = CoCoMem.from_json(llm, json.load(open(f"{ROOT}/states/cocomem/{e5['qid']}.json")))
            m = mems[e5["qid"]]
            u = m.units.get(e5["uid"])
            if u is None:
                return dict(base, y=e5["y"], target=e5["target"], scores=[])
            cid = re.match(r"\[(c\d+)\]", e5["old"]).group(1)
            facts = unit_facts(m, u, cid, e5["new"])
            old = re.sub(r"^\[c\d+\]\s*", "", e5["old"])
            sc = await asyncio.gather(*(p_yes(VERIFY.format(facts=facts, old=old, new=e5["new"], sent=strip_cites(s),
                                                            cand=strip_cites(c))) for c in uniq))
            best = max(range(len(uniq)), key=lambda j: (sc[j], -edit_ratio(s, uniq[j])))
            t = uniq[best]
            return dict(base, y=int(norm(t) != norm(s)), target=t, scores=sc, cands=uniq)

        R = await gather_limited([one(i) for i in range(len(I5))], desc=f"relabel_cv-{name}", every=2000)
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for e in R:
                f.write(json.dumps(e) + "\n")
        src_of = {"copy": 0, "tnet5": 0, "tnet7": 0, "ledger": 0}
        for e in R:
            t = norm(e["target"])
            for k, c in (("copy", e["sent"]), ("tnet5", e["target5"]), ("tnet7", e["target7"]), ("ledger", e.get("t_ledger") or "")):
                if t == norm(c):
                    src_of[k] += 1
                    break
        print(name, len(R), "pos", sum(e["y"] for e in R), "from", json.dumps(src_of),
              "agree_tnet5", sum(e["y"] == e["y5"] for e in R) / len(R), flush=True)


async def _relabel_cv2(a):
    """One commutator with three outcomes (v12).  The candidate set of relabel_cv is extended with the compressor's
    reasoning-mode repair of the sentence from its derivation scope (the global path of the commutation square,
    computed with deliberate reasoning).  Candidates fall into two classes: LOCAL (copy and the two-input proposals),
    which a function of (delta, s) can in principle reproduce, and GLOBAL (repairs that read the unit's facts).
    The verified winner decides the target.  A local winner: the target is that sentence (or the copy).  A global
    winner: the sentence is not determined by (delta, s) alone, and the target is  [LEDGER] <best local candidate>:
    the commutator abstains and still offers its local draft.  One network, one signal, three outcomes."""
    from openai import AsyncOpenAI
    from cocomem import CoCoMem, strip_cites, ledger_call
    from llm import LLM, gather_limited
    llm = LLM("qwen3-8b", 18431, concurrency=96)
    cli = AsyncOpenAI(base_url="http://localhost:18431/v1", api_key="EMPTY", timeout=3600)
    sem = asyncio.Semaphore(160)
    mems = {}
    os.makedirs(a.out, exist_ok=True)

    async def p_yes(prompt):
        async with sem:
            for attempt in range(6):
                try:
                    r = await cli.chat.completions.create(
                        model="qwen3-8b", messages=[{"role": "user", "content": prompt}], max_tokens=1,
                        temperature=0.0, logprobs=True, top_logprobs=10,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    await asyncio.sleep(5 * (attempt + 1))
        py = pn = 0.0
        for t in r.choices[0].logprobs.content[0].top_logprobs:
            w = t.token.strip().lower()
            if w == "yes":
                py += math.exp(t.logprob)
            elif w == "no":
                pn += math.exp(t.logprob)
        return py / (py + pn) if py + pn > 0 else 0.5

    def mem_of(qid):
        if qid not in mems:
            mems[qid] = CoCoMem.from_json(llm, json.load(open(f"{ROOT}/states/cocomem/{qid}.json")))
        return mems[qid]

    for name in ("train", "test"):
        ck = f"{a.out}/{name}.think.jsonl"
        E = [json.loads(l) for l in open(ck if os.path.exists(ck) else f"{a.src}/{name}.jsonl")]
        groups = {}
        for i, e in enumerate(E):
            groups.setdefault((e["qid"], e["uid"], e["old"], e["new"]), []).append(i)

        async def think_group(key, idx):                    # reasoning-mode repair of the whole unit scope
            qid, uid, old, new = key
            m = mem_of(qid)
            u = m.units.get(uid)
            if u is None:
                return
            cid = re.match(r"\[(c\d+)\]", old).group(1)
            K = max(E[i]["k"] for i in idx) + 1
            sents = [""] * K
            for i in idx:
                sents[E[i]["k"]] = E[i]["sent"]
            scope = m._derivation_scope(u, {cid}, sents, [E[i]["k"] for i in idx])
            lines = []
            for c in scope:
                x = m.claims[c]
                lines.append(f"[{c}] ({x['date'][:10]}, {x['speaker']}) {new if c == cid else x['text']}")
            try:
                got = await ledger_call(llm, u["level"], old, new, "\n".join(lines), sents, [E[i]["k"] for i in idx], think=True)
            except Exception:
                got = {}
            for i in idx:
                E[i]["t_think"] = got.get(E[i]["k"]) or E[i]["sent"]

        if not os.path.exists(ck):
            await gather_limited([think_group(k_, v) for k_, v in groups.items()], desc=f"think-{name}", every=200)
            with open(ck, "w") as f:                        # checkpoint: the expensive part
                for e in E:
                    f.write(json.dumps(e) + "\n")

        async def one(i):
            e = E[i]
            s = e["sent"]
            cls, txt = {}, {}
            for c, kind in ((s, "L"), (e.get("target5"), "L"), (e.get("target7"), "L"),
                            (e.get("t_ledger"), "G"), (e.get("t_think"), "G")):
                if c and c != LEDGER and norm(c) and norm(c) not in txt:
                    txt[norm(c)] = c
                    cls[norm(c)] = kind
            prev = {norm(c): v for c, v in zip(e.get("cands") or [], e.get("scores") or [])} if len(e.get("cands") or []) > 1 else {}
            sc = {k_: prev[k_] for k_ in txt if k_ in prev}
            need = [k_ for k_ in txt if k_ not in sc]
            base = dict(e, y10=e["y"], target10=e["target"])
            base.pop("cands", None)
            base.pop("scores", None)
            if len(txt) <= 1:
                return dict(base, y=0, target=s, kind="copy")
            m = mem_of(e["qid"])
            u = m.units.get(e["uid"])
            if u is None:
                return dict(base, y=e["y"], target=e["target"], kind="keep")
            cid = re.match(r"\[(c\d+)\]", e["old"]).group(1)
            facts = unit_facts(m, u, cid, e["new"])
            old = re.sub(r"^\[c\d+\]\s*", "", e["old"])
            got = await asyncio.gather(*(p_yes(VERIFY.format(facts=facts, old=old, new=e["new"], sent=strip_cites(s),
                                                             cand=strip_cites(txt[k_]))) for k_ in need))
            sc.update(dict(zip(need, got)))
            order = lambda ks: max(ks, key=lambda k_: (sc[k_], -edit_ratio(s, txt[k_]), cls[k_] == "L"))
            win = order(list(txt))
            loc = order([k_ for k_ in txt if cls[k_] == "L"])
            if cls[win] == "L":
                t = txt[win]
                return dict(base, y=int(norm(t) != norm(s)), target=t, kind="local", scores=sc)
            return dict(base, y=2, target=f"{LEDGER} {txt[loc]}", kind="global", scores=sc, draft=txt[loc])

        R = await gather_limited([one(i) for i in range(len(E))], desc=f"relabel_cv2-{name}", every=2000)
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            for e in R:
                f.write(json.dumps(e) + "\n")
        print(name, len(R), "copy", sum(e["y"] == 0 for e in R), "rewrite", sum(e["y"] == 1 for e in R),
              "ledger", sum(e["y"] == 2 for e in R),
              "ledger_draft_changed", sum(e["y"] == 2 and norm(e["draft"]) != norm(e["sent"]) for e in R), flush=True)


async def _relabel_cv2(a):
    """One commutator with three outcomes (v12).  The candidate set of relabel_cv is extended with the compressor's
    reasoning-mode repair of the sentence from its derivation scope (the global path of the commutation square,
    computed with deliberate reasoning).  Candidates fall into two classes: LOCAL (copy and the two-input proposals),
    which a function of (delta, s) can in principle reproduce, and GLOBAL (repairs that read the unit's facts).
    The verified winner decides the target.  A local winner: the target is that sentence (or the copy).  A global
    winner: the sentence is not determined by (delta, s) alone, and the target is  [LEDGER] <best local candidate>:
    the commutator abstains and still offers its local draft.  One network, one signal, three outcomes."""
    from openai import AsyncOpenAI
    from cocomem import CoCoMem, strip_cites, ledger_call
    from llm import LLM, gather_limited
    llm = LLM("qwen3-8b", 18431, concurrency=96)
    cli = AsyncOpenAI(base_url="http://localhost:18431/v1", api_key="EMPTY", timeout=3600)
    sem = asyncio.Semaphore(160)
    mems = {}
    os.makedirs(OUT, exist_ok=True)

    async def p_yes(prompt):
        async with sem:
            for attempt in range(6):
                try:
                    r = await cli.chat.completions.create(
                        model="qwen3-8b", messages=[{"role": "user", "content": prompt}], max_tokens=1,
                        temperature=0.0, logprobs=True, top_logprobs=10,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    await asyncio.sleep(5 * (attempt + 1))
        py = pn = 0.0
        for t in r.choices[0].logprobs.content[0].top_logprobs:
            w = t.token.strip().lower()
            if w == "yes":
                py += math.exp(t.logprob)
            elif w == "no":
                pn += math.exp(t.logprob)
        return py / (py + pn) if py + pn > 0 else 0.5

    def mem_of(qid):
        if qid not in mems:
            mems[qid] = CoCoMem.from_json(llm, json.load(open(f"{ROOT}/states/cocomem/{qid}.json")))
        return mems[qid]

    for name in ("train", "test"):
        E = [json.loads(l) for l in open(f"{a.src}/{name}.jsonl")]
        groups = {}
        for i, e in enumerate(E):
            groups.setdefault((e["qid"], e["uid"], e["old"], e["new"]), []).append(i)

        async def think_group(key, idx):                    # reasoning-mode repair of the whole unit scope
            qid, uid, old, new = key
            m = mem_of(qid)
            u = m.units.get(uid)
            if u is None:
                return
            cid = re.match(r"\[(c\d+)\]", old).group(1)
            K = max(E[i]["k"] for i in idx) + 1
            sents = [""] * K
            for i in idx:
                sents[E[i]["k"]] = E[i]["sent"]
            scope = m._derivation_scope(u, {cid}, sents, [E[i]["k"] for i in idx])
            lines = []
            for c in scope:
                x = m.claims[c]
                lines.append(f"[{c}] ({x['date'][:10]}, {x['speaker']}) {new if c == cid else x['text']}")
            try:
                got = await ledger_call(llm, u["level"], old, new, "\n".join(lines), sents, [E[i]["k"] for i in idx], think=True)
            except Exception:
                got = {}
            for i in idx:
                E[i]["t_think"] = got.get(E[i]["k"]) or E[i]["sent"]

        await gather_limited([think_group(k_, v) for k_, v in groups.items()], desc=f"think-{name}", every=200)
        with open(f"{a.out}/{name}.think.jsonl", "w") as f:     # checkpoint: the expensive part
            for e in E:
                f.write(json.dumps(e) + "\n")

        async def one(i):
            e = E[i]
            s = e["sent"]
            cls, txt = {}, {}
            for src_, c, kind in (("copy", s, "L"), ("t5", e.get("target5"), "L"), ("t7", e.get("target7"), "L"),
                                  ("ledger", e.get("t_ledger"), "G"), ("think", e.get("t_think"), "G")):
                if c and c != LEDGER and norm(c) and norm(c) not in txt:
                    txt[norm(c)] = c
                    cls[norm(c)] = kind
            prev = {norm(c): v for c, v in zip(e.get("cands") or [], e.get("scores") or [])} if len(e.get("cands") or []) > 1 else {}
            sc = {k_: prev[k_] for k_ in txt if k_ in prev}
            need = [k_ for k_ in txt if k_ not in sc]
            base = dict(e, y10=e["y"], target10=e["target"])
            if len(txt) <= 1:
                return dict(base, y=0, target=s, kind="copy")
            m = mem_of(e["qid"])
            u = m.units.get(e["uid"])
            if u is None:
                return dict(base, y=e["y"], target=e["target"], kind="keep")
            cid = re.match(r"\[(c\d+)\]", e["old"]).group(1)
            facts = unit_facts(m, u, cid, e["new"])
            old = re.sub(r"^\[c\d+\]\s*", "", e["old"])
            got = await asyncio.gather(*(p_yes(VERIFY.format(facts=facts, old=old, new=e["new"], sent=strip_cites(s),
                                                             cand=strip_cites(txt[k_]))) for k_ in need))
            sc.update(dict(zip(need, got)))
            order = lambda ks: max(ks, key=lambda k_: (sc[k_], -edit_ratio(s, txt[k_]), cls[k_] == "L"))
            win = order(list(txt))
            loc = order([k_ for k_ in txt if cls[k_] == "L"])
            if cls[win] == "L":
                t = txt[win]
                return dict(base, y=int(norm(t) != norm(s)), target=t, kind="local", scores=sc)
            return dict(base, y=2, target=f"{LEDGER} {txt[loc]}", kind="global", scores=sc, draft=txt[loc])

        R = await gather_limited([one(i) for i in range(len(E))], desc=f"relabel_cv2-{name}", every=2000)
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            for e in R:
                f.write(json.dumps(e) + "\n")
        print(name, len(R), "copy", sum(e["y"] == 0 for e in R), "rewrite", sum(e["y"] == 1 for e in R),
              "ledger", sum(e["y"] == 2 for e in R),
              "ledger_draft_changed", sum(e["y"] == 2 and norm(e["draft"]) != norm(e["sent"]) for e in R), flush=True)


async def _relabel_w(a):
    """v20 supervision: witness-aware, commutation-verified targets.

    For every (delta, s) pair of the v10 data (same users, same counterfactual corrections) we attach the sentence's
    derivation witness W (witness.py, computed at write time) and add two witness-conditioned proposals of the
    compressor, used as the same three-input operator T(delta, s, W): non-reasoning and reasoning mode.  The
    reasoning proposal is what makes aggregates over several claims (totals, extremes, orders) reachable: with the
    witness the operands are in the input, so the proposal no longer has to guess them.  All candidates
    {copy, v10 candidates, witness proposals} are scored by the compressor reading the PRIVILEGED corrected inputs of
    the whole unit, v(y) = P(yes | I[c->c'], delta, s, y); the target is argmax v, ties broken by the smallest edit.
    Proposals are generated only where some proposer changes the sentence or the witness contains the corrected claim
    (elsewhere every candidate is the copy, and the copy is the target, as in v10).  The unverified ablation keeps the
    reasoning proposal itself as target (well-formedness-filtered, as the v8 unverified targets)."""
    from cocomem import CoCoMem, strip_cites, cited_ids
    from llm import LLM, gather_limited
    llm = LLM("qwen3-8b", 18431, concurrency=a.conc)
    sem = asyncio.Semaphore(160)
    mems, wits = {}, {}
    os.makedirs(a.out, exist_ok=True)
    from llm import Usage
    cost = {"gen": Usage(), "verify": Usage()}
    cost_path = f"{a.out}/label_cost.json"
    if os.path.exists(cost_path):
        for key, previous in json.load(open(cost_path)).items():
            if key in cost:
                cost[key].calls = previous["calls"]
                cost[key].prompt_tokens = previous["in"]
                cost[key].completion_tokens = previous["out"]

    def save_cost():
        with open(cost_path, "w") as f:
            json.dump({k: v.as_dict() for k, v in cost.items()}, f)

    async def p_yes(prompt):
        async with sem:
            for attempt in range(6):
                try:
                    r = await llm.next_client().chat.completions.create(
                        model="qwen3-8b", messages=[{"role": "user", "content": prompt}], max_tokens=1,
                        temperature=0.0, logprobs=True, top_logprobs=10,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    await asyncio.sleep(5 * (attempt + 1))
        cost["verify"].add(r.usage.prompt_tokens, r.usage.completion_tokens)
        py = pn = 0.0
        for t in r.choices[0].logprobs.content[0].top_logprobs:
            w = t.token.strip().lower()
            if w == "yes":
                py += math.exp(t.logprob)
            elif w == "no":
                pn += math.exp(t.logprob)
        return py / (py + pn) if py + pn > 0 else 0.5

    def mem_of(qid):
        if qid not in mems:
            mems[qid] = CoCoMem.from_json(llm, json.load(open(f"{ROOT}/states/cocomem/{qid}.json")))
            p = f"{ROOT}/states/witness/{qid}.json"
            wits[qid] = json.load(open(p))["units"] if os.path.exists(p) else {}
        return mems[qid], wits[qid]

    stats = {"align_exact": 0, "align_search": 0, "align_cites": 0}

    def witness_of(e, m, wd):
        u = m.units.get(e["uid"])
        v = wd.get(e["uid"])
        ins = set(u["inputs"]) if u else set()
        if v:
            S = [norm(x) for x in v["sents"]]
            k = e["k"]
            if k < len(S) and S[k] == norm(e["sent"]):
                stats["align_exact"] += 1
                return v["witness"][k]
            if norm(e["sent"]) in S:
                stats["align_search"] += 1
                return v["witness"][S.index(norm(e["sent"]))]
        stats["align_cites"] += 1          # unit text differs from the annotated one: declared sources only
        return [c for c in cited_ids(e["sent"]) if c in ins]

    def wlines(m, ids, cid):
        out = []
        for c in ids:
            if c != cid and c in m.claims:
                x = m.claims[c]
                out.append(f"[{c}] ({x['date'][:10]}, {x['speaker']}) {x['text']}")
        return out

    for name in ("train", "test"):
        if os.path.exists(f"{a.out}/{name}.jsonl"):
            print(name, "already verified", flush=True)
            continue
        ck = f"{a.out}/{name}.cand.jsonl"
        if os.path.exists(ck):
            E = [json.loads(l) for l in open(ck)]
        else:
            E = [json.loads(l) for l in open(f"{a.src}/{name}.jsonl")]

            async def cand(e):
                m, wd = mem_of(e["qid"])
                cid = re.match(r"\[(c\d+)\]", e["old"]).group(1)
                W = witness_of(e, m, wd)
                e["W"] = W
                e["wfacts"] = render_wfacts(wlines(m, W, cid))
                old_c = {norm(x) for x in (e["sent"], e.get("target"), e.get("target5"), e.get("target7"), e.get("t_ledger"))
                         if x and x != LEDGER}
                e["active"] = int(cid in W or len(old_c) > 1)
                if not e["active"]:
                    e["tw"] = e["tt"] = e["sent"]
                    return e
                p = wprompt(e)
                tw, tt = await asyncio.gather(llm.chat(p, max_tokens=out_budget(e["sent"]), usage=cost["gen"]),
                                              llm.chat(p, max_tokens=out_budget(e["sent"]) + 3000, think=True, usage=cost["gen"]))
                e["tw"], e["tt"] = first_line(tw) or e["sent"], first_line(tt) or e["sent"]
                return e

            E = await gather_limited([cand(e) for e in E], desc=f"wcand-{name}", every=2000)
            with open(ck + ".tmp", "w") as f:              # checkpoint: the expensive part
                for e in E:
                    f.write(json.dumps(e) + "\n")
            os.replace(ck + ".tmp", ck)
            save_cost()

        async def verify(e):
            s = e["sent"]
            # New proposals need the same sentence-only contract as the v10
            # targets. Entailed commentary is not a usable memory sentence.
            for key in ("tw", "tt"):
                proposal = e.get(key) or s
                if norm(proposal) != norm(s):
                    verdict = await llm.chat(REWRITE.format(a=strip_cites(s), b=strip_cites(proposal)),
                                             max_tokens=4, usage=cost["verify"])
                    if "yes" not in verdict.lower():
                        e[key + "_rejected"] = proposal
                        e[key] = s
            txt = {}
            for c in (s, e.get("target"), e.get("target5"), e.get("target7"), e.get("t_ledger"), e.get("tw"), e.get("tt")):
                if c and c != LEDGER and norm(c) and norm(c) not in txt:
                    txt[norm(c)] = c
            # unverified ablation target: the reasoning proposal, if it is a well-formed version of the sentence
            nv = e.get("tt") or s
            e["target_nv"], e["y_nv"] = nv, int(norm(nv) != norm(s))
            e["y10"], e["target10"] = e["y"], e["target"]
            if len(txt) <= 1:
                e["target"], e["y"], e["wscores"] = s, 0, {}
                return e
            m, _ = mem_of(e["qid"])
            u = m.units.get(e["uid"])
            if u is None:
                e["wscores"] = {}
                return e
            prev = {norm(c): v for c, v in zip(e.get("cands") or [], e.get("scores") or [])}
            cid = re.match(r"\[(c\d+)\]", e["old"]).group(1)
            facts = unit_facts(m, u, cid, e["new"])
            old = re.sub(r"^\[c\d+\]\s*", "", e["old"])
            need = [k_ for k_ in txt if k_ not in prev]
            got = await asyncio.gather(*(p_yes(VERIFY.format(facts=facts, old=old, new=e["new"], sent=strip_cites(s),
                                                             cand=strip_cites(txt[k_]))) for k_ in need))
            sc = {k_: prev[k_] for k_ in txt if k_ in prev} | dict(zip(need, got))
            best = max(txt, key=lambda k_: (sc[k_], -edit_ratio(s, txt[k_])))
            e["target"], e["y"], e["wscores"] = txt[best], int(best != norm(s)), sc
            e["target_src"] = [n for n, c in (("copy", s), ("tw", e.get("tw")), ("tt", e.get("tt")), ("v10", e["target10"]),
                                              ("ledger", e.get("t_ledger"))) if c and norm(c) == best][:1]
            return e

        R = await gather_limited([verify(e) for e in E], desc=f"wverify-{name}", every=2000)
        result_path = f"{a.out}/{name}.jsonl"
        with open(result_path + ".tmp", "w") as f:
            for e in R:
                f.write(json.dumps(e) + "\n")
        os.replace(result_path + ".tmp", result_path)
        src = {}
        for e in R:
            for x in e.get("target_src") or (["copy"] if not e["y"] else []):
                src[x] = src.get(x, 0) + 1
        print(name, len(R), "pos", sum(e["y"] for e in R), "pos_v10", sum(e["y10"] for e in R),
              "pos_nv", sum(e["y_nv"] for e in R), "active", sum(e["active"] for e in R), "src", json.dumps(src),
              "align", json.dumps(stats), flush=True)
        save_cost()


def cmd_data_w(a):
    """Training sets of the v20 student and its ablations from the relabelled pairs (a.src/{train,test}.jsonl).

      tnet20    witness input, verified targets, + retraction pairs        (CoCoMem)
      tnet20nw  NO witness input (two-input prompt), same targets and pairs (- witness)
      tnet20nv  witness input, UNVERIFIED reasoning targets, + retraction   (- verification)
      tnet20nr  witness input, verified targets, NO retraction pairs        (- round trip)

    Retraction (round-trip) pairs.  If y* repairs s for delta = (c -> c'), then the commutative repair of y* for the
    inverse correction delta^-1 = (c' -> c) is EXACTLY s: C(delta^-1(delta(H))) = C(H), and s is the compressor's own
    output for H.  These pairs need no teacher, carry no label noise in their target, and teach the operator to
    recompute the sentence from the correction and its witness rather than from the (possibly stale) sentence: the
    property that keeps errors from accumulating over a sequence of corrections (Theorem 3).  Only changed pairs are
    inverted; the inverse of a copy is a copy."""
    tr = [json.loads(l) for l in open(f"{a.src}/train.jsonl")]
    te = [json.loads(l) for l in open(f"{a.src}/test.jsonl")]
    variants = {"tnet20": ("w", "target", True), "tnet20nw": ("t", "target", True),
                "tnet20nv": ("w", "target_nv", True), "tnet20nr": ("w", "target", False)}

    def ex(e, mode, tkey):
        t = e[tkey]
        x = {k: e[k] for k in ("qid", "uid", "level", "k", "old", "new", "sent", "wfacts", "W") if k in e}
        x["target"], x["y"] = t, int(norm(t) != norm(e["sent"]))
        x["prompt"] = wprompt(x) if mode == "w" else T_PROMPT.format(old=x["old"], new=x["new"], sent=x["sent"])
        return x

    def inverse(x, mode):
        cid = re.match(r"\[(c\d+)\]", x["old"]).group(1)
        old_plain = re.sub(r"^\[c\d+\]\s*", "", x["old"])
        y = dict(x, old=f"[{cid}] {x['new']}", new=old_plain, sent=x["target"], target=x["sent"], rt=1, y=1)
        y["prompt"] = wprompt(y) if mode == "w" else T_PROMPT.format(old=y["old"], new=y["new"], sent=y["sent"])
        return y

    for name, (mode, tkey, rt) in variants.items():
        d = f"{os.path.dirname(a.src.rstrip('/'))}/{name}"
        os.makedirs(d, exist_ok=True)
        T = [ex(e, mode, tkey) for e in tr]
        if rt:
            T += [inverse(x, mode) for x in T if x["y"]]
        with open(f"{d}/train.jsonl", "w") as f:
            for x in T:
                f.write(json.dumps(x) + "\n")
        with open(f"{d}/test.jsonl", "w") as f:              # one held-out set for every variant: verified targets
            for e in te:
                f.write(json.dumps(ex(e, mode, "target")) + "\n")
        print(name, "train", len(T), "pos", sum(x["y"] for x in T), "retraction", sum(bool(x.get("rt")) for x in T),
              "test", len(te), flush=True)


async def _eval(a):
    from llm import LLM, gather_limited
    from cfgen import SAME
    from cocomem import strip_cites
    E = [json.loads(l) for l in open(f"{OUT}/test.jsonl")]
    T = LLM(a.served, a.port, concurrency=192)
    J = LLM("qwen3-8b", 18431, concurrency=128)

    async def one(e):
        out = first_line(await T.chat(prompt_text(e), max_tokens=out_budget(e["sent"])))
        if out.startswith(LEDGER):
            return {"changed": True, "ledger": True, "good": None, "out": out}
        changed = norm(out) != norm(e["sent"])
        good = None
        if e["y"] and changed:
            good = norm(out) == norm(e["target"]) or "no" not in (await J.chat(
                SAME.format(a=strip_cites(e["target"]), b=strip_cites(out)), max_tokens=4)).lower()
        return {"changed": changed, "good": good, "out": out}

    R = await gather_limited([one(e) for e in E], desc="eval", every=1000)
    tp = sum(r["changed"] and e["y"] for r, e in zip(R, E))
    fp = sum(r["changed"] and not e["y"] for r, e in zip(R, E))
    fn = sum((not r["changed"]) and e["y"] for r, e in zip(R, E))
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res = {"p": p, "r": rc, "f1": 2 * p * rc / max(1e-9, p + rc), "rewrite_ok": sum(bool(r["good"]) for r in R) / max(1, tp),
           "exact_commute": sum((bool(r["good"]) if e["y"] else not r["changed"]) for r, e in zip(R, E)) / len(E)}
    for lev in ("note", "digest", "profile"):
        ii = [i for i, e in enumerate(E) if e["level"] == lev]
        tp_ = sum(R[i]["changed"] and E[i]["y"] for i in ii); fp_ = sum(R[i]["changed"] and not E[i]["y"] for i in ii)
        fn_ = sum((not R[i]["changed"]) and E[i]["y"] for i in ii)
        res[lev] = {"p": tp_ / max(1, tp_ + fp_), "r": tp_ / max(1, tp_ + fn_)}
    if any(e["y"] == 2 for e in E):      # sufficiency decisions of the three-way commutator
        lp = sum(bool(r.get("ledger")) and e["y"] == 2 for r, e in zip(R, E))
        res["ledger_p"] = lp / max(1, sum(bool(r.get("ledger")) for r in R))
        res["ledger_r"] = lp / max(1, sum(e["y"] == 2 for e in E))
        res["ledger_rate"] = sum(bool(r.get("ledger")) for r in R) / len(R)
    print(json.dumps(res), flush=True)
    json.dump({"res": res, "out": R}, open(f"{ROOT}/results/tnet_eval_{a.served}{a.suffix}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", default=f"{ROOT}/data/cf_dev.jsonl,{ROOT}/data/cf_train.jsonl")
    ap.add_argument("--test", default=f"{ROOT}/data/cf_test.jsonl")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="Training initialization, dropout and shuffle seed; evaluation examples remain fixed")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--tok_budget", type=int, default=8192)
    ap.add_argument("--qlora", action="store_true", help="train: 4-bit NF4 frozen base (same LoRA recipe), merged into the bf16 base")
    ap.add_argument("--model", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--served", default="tnet")
    ap.add_argument("--port", type=int, default=18432)
    ap.add_argument("--src", default=OUT)
    ap.add_argument("--prompt", type=int, default=1)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--conc", type=int, default=96)
    a = ap.parse_args()
    BASE, OUT = a.base, a.out
    PROMPT = prompt3 if a.prompt == 3 else prompt_of
    if a.cmd == "data":
        cmd_data(a)
    elif a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "score":
        cmd_score(a)
    elif a.cmd == "relabel3":
        asyncio.run(_relabel3(a))
    elif a.cmd == "relabel":
        asyncio.run(_relabel(a))
    elif a.cmd == "relabel_cv":
        asyncio.run(_relabel_cv(a))
    elif a.cmd == "relabel_cv2":
        asyncio.run(_relabel_cv2(a))
    elif a.cmd == "relabel_cv2":
        asyncio.run(_relabel_cv2(a))
    elif a.cmd == "relabel_w":
        asyncio.run(_relabel_w(a))
    elif a.cmd == "data_w":
        cmd_data_w(a)
    elif a.cmd == "eval":
        asyncio.run(_eval(a))
