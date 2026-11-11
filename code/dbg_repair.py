import asyncio, json, sys
import cocomem, prompts
from cocomem import CoCoMem
from data import load_lme
from llm import LLM, Usage, parse_json
from scenarios import corr_session, load_corr
qid = sys.argv[1]
items = {x["qid"]: x for x in load_lme()}; corr = load_corr()
llm = LLM("qwen3-8b", 18431)
orig_chat = llm.chat
async def spy(pr, **kw):
    r = await orig_chat(pr, **kw)
    if isinstance(pr, str) and pr.startswith("You maintain a memory"):
        print("----PROMPT tail:", pr[:300].replace("\n"," | "), "\n----OUT:", r[:600])
    return r
llm.chat = spy
async def main():
    m = CoCoMem.from_json(llm, json.load(open(f"/data/yaokang/corrmem/states/cocomem/{qid}.json")))
    m.set_lineage("coarse")
    info = await m.correct(corr_session(items[qid], corr[qid], "affect"), strategy="coarse", usage=Usage())
    print(info)
asyncio.run(main())
