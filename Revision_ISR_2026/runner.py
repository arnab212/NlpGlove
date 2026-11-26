# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
runner.py - run one continual-learning configuration on one Experiment-4
participant and return every metric the revision needs.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import f1_score

from cl_core import CLConfig, ContinualLearner, linear_cka
from common import (BASE_MODEL, PARTICIPANTS, load_base_dataset, load_exp1,
                    load_participant)

N_ADAPT = 140          # rows 1-140 adaptation, 141-210 held-out test (paper protocol)
N_EARLY = 35           # "early" items used for the retention / forgetting test

_DATA = {}


def data():
    """Load everything once."""
    if not _DATA:
        _DATA["users"] = {u: load_participant(u) for u in PARTICIPANTS}
        _DATA["exp1"] = load_exp1()
        base = load_base_dataset().sample(frac=1.0, random_state=123).reset_index(drop=True)
        _DATA["base_eval"] = base.iloc[:300].reset_index(drop=True)     # forgetting probe
        _DATA["anchor_pool"] = base.iloc[300:].reset_index(drop=True)   # anchor replay pool
    return _DATA


def run_one(cfg: CLConfig, user: str, cross_users=True, drift=True) -> dict:
    D = data()
    df = D["users"][user]
    adapt, test = df.iloc[:N_ADAPT], df.iloc[N_ADAPT:]
    ex1, beval = D["exp1"], D["base_eval"]
    t0 = time.time()

    L = ContinualLearner(cfg, BASE_MODEL, anchor_pool=D["anchor_pool"])

    # ---- before adaptation
    base_test, _ = L.evaluate(test.prompt, test.label)
    gen_before, _ = L.evaluate(ex1.prompt, ex1.labels)
    bd_before, _ = L.evaluate(beval.prompt, beval.label)
    early_before, _ = L.evaluate(adapt.prompt[:N_EARLY], adapt.label[:N_EARLY])
    cross_before = {}
    if cross_users:
        for o in D["users"]:
            if o != user:
                t = D["users"][o].iloc[N_ADAPT:]
                cross_before[o], _ = L.evaluate(t.prompt, t.label)
    probe_texts = list(ex1.prompt) + list(beval.prompt)
    emb_before = L.embeddings(probe_texts) if drift else None

    # ---- adaptation phase (prequential: predict, then learn from feedback)
    online = []
    for text, lab in zip(adapt.prompt, adapt.label):
        _, ok = L.observe(text, lab)
        online.append(ok)

    # ---- after adaptation (model frozen)
    test_acc, test_pred = L.evaluate(test.prompt, test.label)
    gen_after, _ = L.evaluate(ex1.prompt, ex1.labels)
    bd_after, _ = L.evaluate(beval.prompt, beval.label)
    early_after, _ = L.evaluate(adapt.prompt[:N_EARLY], adapt.label[:N_EARLY])
    seen_after, _ = L.evaluate(adapt.prompt, adapt.label)
    cross_after = {}
    for o in cross_before:
        t = D["users"][o].iloc[N_ADAPT:]
        cross_after[o], _ = L.evaluate(t.prompt, t.label)

    row = cfg.as_row()
    row.update({
        "user": user,
        "base_test_acc": base_test,
        "test_acc": test_acc,
        "gain_pp": 100 * (test_acc - base_test),
        "test_macro_f1": f1_score(list(test.label), test_pred, average="macro"),
        "online_acc": float(np.mean(online)),
        "online_acc_last40": float(np.mean(online[-40:])),
        "n_updates": L.n_updates,
        "n_gated": L.n_gated,
        "final_buffer_len": len(L.buffer.items),
        "exp1_general_before": gen_before,
        "exp1_general_after": gen_after,
        "forget_exp1_pp": 100 * (gen_before - gen_after),
        "basedata_before": bd_before,
        "basedata_after": bd_after,
        "forget_basedata_pp": 100 * (bd_before - bd_after),
        "early_items_before": early_before,
        "early_items_after": early_after,
        "seen_items_after": seen_after,
        "cross_user_gain_pp": (100 * float(np.mean([cross_after[o] - cross_before[o] for o in cross_before]))
                               if cross_before else np.nan),
        "loss_first5": float(np.mean(L.losses[:5])) if L.losses else np.nan,
        "loss_last5": float(np.mean(L.losses[-5:])) if L.losses else np.nan,
        "seconds": time.time() - t0,
    })
    if drift:
        row["cka_backbone_vs_base"] = linear_cka(emb_before, L.embeddings(probe_texts))
    if cross_users:                      # fixed column set -> consistent CSV rows
        for o in PARTICIPANTS:
            row[f"cross_{o}"] = cross_after.get(o, np.nan)
    return row
