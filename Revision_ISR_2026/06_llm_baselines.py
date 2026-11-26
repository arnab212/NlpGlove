# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
06_llm_baselines.py - fair online/offline comparison (R1 #1, R2 #7):

  * DeepSeek (zero-shot, API) on the Exp-4 held-out sets -> lets you put
    DeepSeek next to the personalised DistilBERT on the SAME 70 prompts per
    participant (the paper currently has no DeepSeek number for Exp 4).
  * A local, offline LLM (GGUF via llama-cpp-python, grammar-constrained to the
    7 labels) on Exp 1, Exp 2 transcripts and Exp-4 test sets. Shows whether
    "offline" must mean "small classifier", and what it costs in latency.

Examples
  set DEEPSEEK_API_KEY=sk-...           (Windows cmd)   / $env:DEEPSEEK_API_KEY="sk-..." (PowerShell)
  python 06_llm_baselines.py --backend deepseek --model deepseek-chat --sets exp4
  python 06_llm_baselines.py --backend deepseek --model deepseek-reasoner --sets exp4
  python 06_llm_baselines.py --backend gguf --gguf ..\\models\\mistral-7b-instruct-v0.2.Q4_K_M.gguf
  python 06_llm_baselines.py --backend gguf --gguf D:\\models\\Qwen2.5-7B-Instruct-Q4_K_M.gguf

Results are cached per (backend, prompt) in outputs/llm_cache_<backend>.jsonl,
so an interrupted run resumes without paying for the same call twice.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import OUT, PARTICIPANTS, load_exp1, load_exp2, load_participant


def build_sets(which):
    sets = {}
    if "exp1" in which:
        e1 = load_exp1()
        sets["exp1"] = list(zip(e1.prompt, e1.labels, ["-"] * len(e1)))
    if "exp2" in which:
        e2 = load_exp2()
        sets["exp2"] = list(zip(e2.prompt, e2.labels, e2.user.astype(str)))
    if "exp4" in which:
        rows = []
        for u in PARTICIPANTS:
            t = load_participant(u).iloc[140:]
            rows += [(p, {l}, u) for p, l in zip(t.prompt, t.label)]
        sets["exp4"] = rows
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["deepseek", "gguf"], required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--sets", nargs="+", default=["exp1", "exp2", "exp4"])
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N prompts per set")
    args = ap.parse_args()

    from llm_backends import DeepSeekBackend, LlamaCppBackend
    be = DeepSeekBackend(args.model) if args.backend == "deepseek" else LlamaCppBackend(args.gguf)
    cache_f = OUT / f"llm_cache_{be.name.replace(':', '_').replace('/', '_')}.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            cache[j["prompt"]] = j

    rows = []
    for sname, items in build_sets(args.sets).items():
        if args.limit:
            items = items[:args.limit]
        for i, (prompt, labels, grp) in enumerate(items):
            if prompt not in cache:
                r = be(prompt)
                r["prompt"] = prompt
                cache[prompt] = r
                with open(cache_f, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
            r = cache[prompt]
            rows.append({"set": sname, "group": grp, "prompt": prompt, "pred": r["label"],
                         "ok": r["label"] in labels, "parse_fail": r["label"] is None,
                         "latency_s": r["latency_s"]})
            if (i + 1) % 20 == 0:
                print(f"{sname}: {i+1}/{len(items)}  acc so far {np.mean([x['ok'] for x in rows if x['set']==sname]):.3f}",
                      flush=True)
    R = pd.DataFrame(rows)
    tag = be.name.replace(":", "_")
    R.to_csv(OUT / f"llm_baseline_{tag}.csv", index=False)
    md = [f"# {be.name}\n", "| set | n | accuracy | parse failures | median latency (s) |", "|---|---|---|---|---|"]
    for s, g in R.groupby("set"):
        md.append(f"| {s} | {len(g)} | {g.ok.mean():.3f} | {g.parse_fail.sum()} | {g.latency_s.median():.2f} |")
    if (R.set == "exp4").any():
        md.append("\nExp 4 per participant (compare with base / personalised DistilBERT in table2_recomputed.csv):")
        for u, g in R[R.set == "exp4"].groupby("group"):
            md.append(f"- {u}: {g.ok.mean():.3f}")
    (OUT / f"llm_baseline_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
