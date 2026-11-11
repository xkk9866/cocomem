"""Throughput probe: claim extraction on N sessions."""
import asyncio
import sys
import time

from data import load_lme, format_session
from llm import LLM, Usage, gather_limited, parse_json
import prompts


async def main(n):
    items = load_lme()
    seen, sess = set(), []
    for x in items:
        for s in x["sessions"]:
            if s["hash"] not in seen:
                seen.add(s["hash"])
                sess.append(s)
    sess = sess[:n]
    llm = LLM("qwen3-8b", 18431, concurrency=128, cache_name="tput_probe")
    u = Usage()
    t0 = time.time()
    outs = await gather_limited([llm.chat(prompts.EXTRACT.format(session=format_session(s["turns"], s["date"])),
                                          max_tokens=900, usage=u) for s in sess], desc="extract", every=25)
    el = time.time() - t0
    from claims import parse_claims; pj = [{"claims": parse_claims(o)} for o in outs]
    ok = sum(p.get("claims") is not None for p in pj)
    nc = sum(len(p.get("claims") or []) for p in pj)
    print(f"n={n} time={el:.1f}s in={u.prompt_tokens} out={u.completion_tokens} in/s={u.prompt_tokens / el:.0f} out/s={u.completion_tokens / el:.0f} parsed={ok} claims={nc}")
    print(outs[0][:800]); print(outs[5][:800])


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1])))
