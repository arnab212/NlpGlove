# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
07_summarise_ablations.py - turn outputs/ablations_*.csv into paper-ready
tables (markdown + LaTeX) and figures.

For each configuration (mean +/- SD over participants x seeds):
  base -> adapted held-out accuracy, gain, #updates, forgetting on the base
  training distribution and on Exp-1 prompts, retention of early items,
  backbone drift (linear CKA vs. the base model), cross-user transfer.

Also: paired item-level McNemar is not possible from aggregate rows, so the
script reports a participant-level exact sign test of "config beats paper".

Run: python 07_summarise_ablations.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import OUT, PARTICIPANTS

METRICS = [("test_acc", "Adapted acc", "{:.3f}"), ("gain_pp", "Gain (pp)", "{:+.1f}"),
           ("n_updates", "Updates", "{:.0f}"), ("forget_basedata_pp", "Forget base (pp)", "{:+.1f}"),
           ("forget_exp1_pp", "Forget Exp1 (pp)", "{:+.1f}"), ("early_items_after", "Early-item acc", "{:.3f}"),
           ("cka_backbone_vs_base", "CKA vs base", "{:.3f}"), ("cross_user_gain_pp", "Cross-user gain (pp)", "{:+.1f}")]


def sign_test_vs(df, cfg, ref="paper"):
    a = df[df.name == cfg].groupby("user").test_acc.mean()
    b = df[df.name == ref].groupby("user").test_acc.mean()
    d = (a - b).dropna()
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    k = (d > 0).sum()
    n = len(d)
    from math import comb
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)


def table(df, title):
    order = list(dict.fromkeys(df.name))
    g = df.groupby("name")
    lines = [f"## {title}\n",
             "| Config | runs | " + " | ".join(m[1] for m in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    tex = ["\\begin{tabular}{l" + "c" * len(METRICS) + "}", "\\toprule",
           "Config & " + " & ".join(m[1] for m in METRICS) + " \\\\", "\\midrule"]
    for name in order:
        x = g.get_group(name)
        cells, tcells = [], []
        for col, _, fmt in METRICS:
            if col not in x or x[col].isna().all():
                cells.append("-"); tcells.append("-")
                continue
            m, s = x[col].mean(), x[col].std(ddof=1) if len(x) > 1 else 0.0
            cells.append(f"{fmt.format(m)} ± {fmt.format(s).lstrip('+')}")
            tcells.append(f"{fmt.format(m)}$\\pm${fmt.format(s).lstrip('+')}")
        lines.append(f"| {name} | {len(x)} | " + " | ".join(cells) + " |")
        tex.append(name.replace("_", "\\_") + " & " + " & ".join(tcells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    if "paper" in order:
        lines.append("\nExact sign test across participants, config vs 'paper' (two-sided p):")
        for name in order:
            if name != "paper":
                lines.append(f"- {name}: p = {sign_test_vs(df, name):.3f}")
    return lines, tex


def figures(df, suite):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    order = list(dict.fromkeys(df.name))
    g = df.groupby("name")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, lab) in zip(axes, [("test_acc", "Held-out accuracy (adapted)"),
                                     ("forget_basedata_pp", "Forgetting on base data (pp)"),
                                     ("cka_backbone_vs_base", "Backbone CKA vs base")]):
        means = [g.get_group(n)[col].mean() for n in order]
        sds = [g.get_group(n)[col].std(ddof=1) for n in order]
        ax.barh(range(len(order)), means, xerr=sds, color="#4C72B0", alpha=.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_ablation_{suite}.png", dpi=200)
    plt.close(fig)
    if suite == "noise":
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in ("dual", "classifier_only", "lora", "dual_gate"):
            xs, ys, es = [], [], []
            for nz in (0, 10, 20, 30):
                n = f"noise{nz}_{method}"
                if n in g.groups:
                    xs.append(nz); ys.append(g.get_group(n).test_acc.mean()); es.append(g.get_group(n).test_acc.std(ddof=1))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)
        ax.set_xlabel("Wrong user corrections (%)")
        ax.set_ylabel("Held-out accuracy after adaptation")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_noise_robustness.png", dpi=200)
        plt.close(fig)


def main():
    md, tex = ["# Ablation summary (auto-generated by 07_summarise_ablations.py)\n",
               "Mean ± SD over participants x seeds. Forgetting = accuracy drop on 300 prompts of the base "
               "fine-tuning set / on the 96 Exp-1 prompts (positive = forgot). CKA = linear CKA of the backbone "
               "[CLS] representation vs the base model on 396 probe prompts (1 = unchanged).\n"], []
    for f in sorted(OUT.glob("ablations_*.csv")):
        suite = f.stem.replace("ablations_", "")
        df = pd.read_csv(f)
        df = df[df.user.isin(list(PARTICIPANTS))]      # only the 4 paper participants
        l, t = table(df, f"Suite: {suite}")
        md += l + [""]
        tex += [f"% suite {suite}"] + t + [""]
        figures(df, suite)
    (OUT / "ablation_summary.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "ablation_tables.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
