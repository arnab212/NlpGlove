# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
04_exp2_adaptation.py - the head-to-head Reviewer 1 is really asking for:
"can the offline, adaptive DistilBERT close the gap to zero-shot DeepSeek on
the SAME real spoken commands?"  Uses the 200 Whisper transcripts of Exp 2.

Reference label of an utterance = the user's explicit correction if one was
given for either model, otherwise the confirmed prediction. Accuracy is scored
with the accepted-label set (as in Exp 1), so DeepSeek / DistilBERT numbers are
directly comparable with the logged ones.

  A. Leave-one-participant-out (LOPO), population adaptation:
       continual learning over the other 19 users' 190 commands, then frozen
       evaluation on the held-out user's 10 commands.
  B. LOPO standard fine-tuning (3 epochs on the 190 commands) - tells you
       whether the gap is a distribution-shift problem (training prompts were
       short and templated; real speech is longer and conversational).
  C. Sequential deployment: one model, users 1..20 in the order they were
       tested, prequential accuracy (predict first, then learn).

Writes outputs/exp2_adaptation.csv and outputs/exp2_adaptation.md
Run: python 04_exp2_adaptation.py [--seeds 0 1 2] [--parts A B C]
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cross_entropy

from cl_core import DEVICE, CLConfig, ContinualLearner, load_base, predict, set_seed
from common import BASE_MODEL, LABEL2ID, LABELS, OUT, load_exp2

CFG = CLConfig(name="paper")          # best configuration in 03 --suite lr


def ref_label(r):
    labs = sorted(r.labels)
    # prefer a label that came from a correction (it is the user's own statement)
    corr = [l for l in labs if not ((r.bert_ok and l == r.bert_pred) or (r.deepseek_ok and l == r.deepseek_pred))]
    if corr:
        return corr[0]
    if r.deepseek_ok:
        return r.deepseek_pred
    if r.bert_ok:
        return r.bert_pred
    return labs[0] if labs else None


def acc_set(preds, sets):
    return float(np.mean([p in s for p, s in zip(preds, sets)]))


def finetune(train_texts, train_labels, seed, epochs=3, lr=3e-5, bs=16):
    set_seed(seed)
    tok, base = load_base(BASE_MODEL)
    m = copy.deepcopy(base).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    idx = np.arange(len(train_texts))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        m.train()
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            enc = tok([train_texts[j] for j in b], return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(DEVICE)
            y = torch.tensor([LABEL2ID[train_labels[j]] for j in b], device=DEVICE)
            loss = cross_entropy(m(**enc).logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return tok, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C"])
    args = ap.parse_args()

    e2 = load_exp2()
    e2["ref"] = [ref_label(r) for r in e2.itertuples()]
    e2 = e2.dropna(subset=["ref"]).reset_index(drop=True)
    tok, base = load_base(BASE_MODEL)
    ids, _ = predict(base, tok, list(e2.prompt))
    e2["base_pred_now"] = [LABELS[i] for i in ids]
    rows = []
    md = ["# Experiment 2 - can the adaptive offline model close the gap? (04_exp2_adaptation.py)\n",
          f"- utterances: {len(e2)} from 20 participants (Whisper transcripts)",
          f"- logged DistilBERT (during study) accuracy vs accepted set: {acc_set(e2.bert_pred, e2.labels):.3f}",
          f"- base checkpoint re-run now: {acc_set(e2.base_pred_now, e2.labels):.3f} "
          "(should equal the line above - if not, a different checkpoint was used in the study)",
          f"- logged DeepSeek accuracy vs accepted set: {acc_set(e2.deepseek_pred, e2.labels):.3f}\n"]

    for seed in args.seeds:
        if "A" in args.parts or "B" in args.parts:
            for u in range(1, 21):
                tr, te = e2[e2.user != u], e2[e2.user == u]
                if "A" in args.parts:
                    L = ContinualLearner(replace(CFG, seed=seed), BASE_MODEL)
                    for t, l, s_ in zip(tr.prompt, tr.ref, tr.labels):
                        L.observe(t, l, accepted=s_)
                    a, _ = L.evaluate(te.prompt, list(te.labels))
                    rows.append({"part": "A_lopo_continual", "seed": seed, "user": u, "acc": a,
                                 "updates": L.n_updates})
                if "B" in args.parts:
                    tk, m = finetune(list(tr.prompt), list(tr.ref), seed)
                    pid, _ = predict(m, tk, list(te.prompt))
                    rows.append({"part": "B_lopo_finetune", "seed": seed, "user": u,
                                 "acc": acc_set([LABELS[i] for i in pid], list(te.labels))})
                rows.append({"part": "base_distilbert", "seed": seed, "user": u,
                             "acc": acc_set(te.base_pred_now, list(te.labels))})
                rows.append({"part": "deepseek_logged", "seed": seed, "user": u,
                             "acc": acc_set(te.deepseek_pred, list(te.labels))})
                print(f"seed {seed} user {u:2d}: " + "  ".join(
                    f"{r['part']}={r['acc']:.2f}" for r in rows[-4:] if r["user"] == u), flush=True)
        if "C" in args.parts:
            L = ContinualLearner(replace(CFG, seed=seed), BASE_MODEL)
            for u in range(1, 21):
                te = e2[e2.user == u]
                hits = []
                for t, l, s in zip(te.prompt, te.ref, te.labels):
                    pred, ok = L.observe(t, l, accepted=s)   # predict BEFORE learning from it
                    hits.append(ok)
                rows.append({"part": "C_sequential_prequential", "seed": seed, "user": u,
                             "acc": float(np.mean(hits)), "updates": L.n_updates})
            print(f"seed {seed} sequential done, total updates {L.n_updates}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "exp2_adaptation.csv", index=False)
    md.append("| Condition | mean acc over users | per-user median | users >= DeepSeek |\n|---|---|---|---|")
    ds = R[R.part == "deepseek_logged"].groupby("user").acc.mean()
    for part, g in R.groupby("part"):
        pu = g.groupby("user").acc.mean()
        md.append(f"| {part} | {pu.mean():.3f} | {pu.median():.2f} | {(pu >= ds.reindex(pu.index)).sum()}/20 |")
    if (R.part == "C_sequential_prequential").any():
        c = R[R.part == "C_sequential_prequential"].groupby("user").acc.mean()
        md.append(f"\nSequential deployment, first 10 users vs last 10 users: {c.iloc[:10].mean():.3f} -> "
                  f"{c.iloc[10:].mean():.3f} (DeepSeek on the same users: {ds.iloc[:10].mean():.3f} -> "
                  f"{ds.iloc[10:].mean():.3f})")
    (OUT / "exp2_adaptation.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
