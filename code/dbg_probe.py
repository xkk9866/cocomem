import json, glob, sys
sys.path.insert(0, ".")
from scenarios import load_corr
corr = load_corr()
R = "/data/yaokang/corrmem/results/main/spec"
k = 0
for f in sorted(glob.glob(R + "/*.json")):
    d = json.load(open(f))
    for a in d["answers"]:
        if a["scenario"] == "S2" and a["qkey"] == "probe" and a.get("label_qwen3-8b") is False and k < 6:
            k += 1
            rec = corr[d["qid"]]["unrel"]
            print("=====", d["qid"], "| corr msg:", rec["correction_message"][:200])
            print("Q:", a["question"], "| gold:", a["gold"], "| hyp:", a["hyp"][:150])
            print("repair:", {x: d["repair"]["S2"].get(x) for x in ("revised", "closure", "rejected")})
            print("full:", a["full"][:300].replace("\n", " "))
