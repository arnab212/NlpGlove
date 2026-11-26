# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
11_results_figures.py - redraws the Results figures that must change, in the style of the
original figures (DistilBERT = blue, DeepSeek = orange, serif fonts, bold axis labels,
italic panel titles, panel letters a), b), ...).

Run AFTER 10_results_numbers.py (it reads that script's CSV outputs):
    python 11_results_figures.py                     # length panel as quartile bins (recommended)
    python 11_results_figures.py --length-panel fit  # length panel as in the original (quadratic fit)

READ-ONLY with respect to the study data. Writes into Revision_ISR_2026/outputs/figs_results/:
    Fig_Exp2_combined_revised.pdf/.png   replaces Combined_Giga_Figure_V4.pdf
    Fig_Exp2_PRF_revised.pdf/.png        replaces PRF_Both_V1.pdf
    Fig_Exp4_personalisation_revised.pdf/.png   replaces P_study1.pdf
    Fig_Exp4_classes_cross_revised.pdf/.png     replaces P_study2.pdf
What changes and why is listed in outputs/figs_results/README.txt.
"""
from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec

from common import LABELS, OUT, PAPER_IDS, PARTICIPANTS, STUDY_DIR, wilson

ap = argparse.ArgumentParser()
ap.add_argument("--length-panel", choices=["bins", "fit"], default="bins")
args = ap.parse_args()

FIG = OUT / "figs_results"
FIG.mkdir(exist_ok=True)

# ----------------------------------------------------------------- style (as the originals)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 12, "axes.titleweight": "normal",
    "axes.labelsize": 12, "axes.labelweight": "bold",
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
})
BERT = "#4a7ebb"          # DistilBERT blue   (original palette)
BERT_LIGHT = "#7fa7e9"    # DistilBERT before adaptation (lighter blue, original palette)
BERT_DARK = "#3a5f9e"     # DistilBERT after adaptation  (darker blue, original palette)
DS = "#ff9900"            # DeepSeek orange   (original palette)
BOX = {"DistilBERT": "#809fe0", "DeepSeek": "#ffad65"}
CM_ORDER = ["grasp", "loosen", "tight_grasp", "open_hand", "pinch_grasp", "gentle_grasp", "tighten"]
NICE = {l: l.replace("_", " ") for l in LABELS}
P_COL = {"P1": "#2ca02c", "P2": "#9467bd", "P3": "#8c564b", "P4": "#555555"}   # no blue/orange: those mean DistilBERT/DeepSeek


def letter(ax, s, x=-0.08, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=17, fontweight="bold", fontstyle="italic",
            va="bottom", ha="left")


def title(ax, s):
    ax.set_title(s, fontstyle="italic", fontsize=11)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("written", FIG / f"{name}.pdf")


# ========================================================================= Experiment 2
E2 = pd.read_csv(OUT / "exp2_items.csv")
fig = plt.figure(figsize=(12, 15.5))
gs = gridspec.GridSpec(3, 2, height_ratios=[1.15, 0.85, 1.05], hspace=0.62, wspace=0.28)

# a) confusion matrices (pooled over participants; rows = participant-confirmed class)
gsa = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :], width_ratios=[1, 1, 0.05, 0.12, 0.05], wspace=0.08)
axs = [fig.add_subplot(gsa[0]), fig.add_subplot(gsa[1])]
cax = [fig.add_subplot(gsa[2]), fig.add_subplot(gsa[4])]   # gsa[3] is an empty gap so the tick labels of both colour bars are visible
for k, (pred, gold, cmap) in enumerate((("bert_pred", "bert_gold", "Blues"), ("ds_pred", "ds_gold", "Oranges"))):
    s = E2[E2[gold].notna()]
    cm = pd.crosstab(pd.Categorical(s[gold], CM_ORDER), pd.Categorical(s[pred], CM_ORDER), dropna=False)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=axs[k], cbar_ax=cax[k], square=True,
                xticklabels=[NICE[l] for l in CM_ORDER], yticklabels=[NICE[l] for l in CM_ORDER] if k == 0 else False,
                annot_kws={"size": 9})
    axs[k].set_xlabel("Predicted class"); axs[k].set_ylabel("")
    axs[k].set_title(["DistilBERT", "DeepSeek"][k], fontstyle="italic", fontsize=11)
    axs[k].tick_params(axis="x", rotation=90)
axs[0].set_ylabel("Confirmed or corrected class")
letter(axs[0], "a)", x=-0.32)

# b) per-participant UCIM distribution
ax = fig.add_subplot(gs[1, 0])
pu = E2.groupby("user")[["bert_yes", "ds_yes"]].mean()
long = pd.DataFrame({"model": ["DistilBERT"] * 20 + ["DeepSeek"] * 20,
                     "ucim": list(pu.bert_yes) + list(pu.ds_yes)})
sns.boxplot(data=long, x="model", y="ucim", hue="model", palette=BOX, ax=ax, width=0.5, legend=False,
            boxprops={"edgecolor": "#0051ff", "linewidth": 1.5}, medianprops={"color": "#fd7600", "linewidth": 2},
            showfliers=False)
sns.stripplot(data=long, x="model", y="ucim", ax=ax, color="#111111", alpha=0.8, size=5, jitter=0.12)
ax.set_xlabel(""); ax.set_ylabel("User-confirmed intent match")
ax.set_ylim(0.2, 1.05); ax.grid(axis="y", alpha=0.4, linestyle="--")
title(ax, "Per-participant UCIM (20 participants, 10 commands each)")
letter(ax, "b)", x=-0.2)

# c) accuracy against command length
ax = fig.add_subplot(gs[1, 1])
if args.length_panel == "bins":
    E2["bin"] = pd.qcut(E2.chars, 4, duplicates="drop")
    cats = list(E2["bin"].cat.categories)
    xs = np.arange(len(cats))
    for k, (pred, gold, col, lab) in enumerate((("bert_pred", "bert_gold", BERT, "DistilBERT"),
                                                ("ds_pred", "ds_gold", DS, "DeepSeek"))):
        acc, lo, hi = [], [], []
        for c in cats:
            g = E2[(E2["bin"] == c) & E2[gold].notna()]
            kk, nn = int((g[pred] == g[gold]).sum()), len(g)
            a, (l_, h_) = kk / nn, wilson(kk, nn)
            acc.append(a); lo.append(a - l_); hi.append(h_ - a)
        ax.errorbar(xs + (k - 0.5) * 0.12, acc, yerr=[lo, hi], color=col, marker="o", ms=6, lw=2,
                    ls="--", capsize=4, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(math.ceil(c.left)) + 1}–{int(c.right)}" for c in cats])
    ax.set_xlabel("Command length (characters, quartiles)")
    title(ax, "Accuracy by command length (95% Wilson intervals)")
else:  # the original analysis (short_long_analysis.py): quadratic fit, commands <= 65 characters
    from scipy.stats import t as tdist
    for pred, gold, col, lab in (("bert_pred", "bert_gold", BERT, "DistilBERT"), ("ds_pred", "ds_gold", DS, "DeepSeek")):
        s = E2[E2[gold].notna() & (E2.chars <= 65)].copy()
        s["ok"] = (s[pred] == s[gold]).astype(int)
        agg = s.groupby("chars").ok.mean()
        x, y = agg.index.values.astype(float), agg.values
        z = np.polyfit(x, y, 2); pf = np.poly1d(z)
        xf = np.linspace(x.min(), x.max(), 300); yf = pf(xf)
        mse = np.mean((y - pf(x)) ** 2); dof = max(1, len(x) - 3)
        se = np.sqrt(mse * (1 / len(x) + (xf - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)))
        ax.plot(xf, yf, "--", color=col, lw=2.5, label=f"{lab} fit")
        ax.fill_between(xf, yf - tdist.ppf(.975, dof) * se, yf + tdist.ppf(.975, dof) * se, color=col, alpha=0.2)
    ax.set_xlabel("Sentence length (characters)")
    title(ax, "Accuracy vs sentence length (commands up to 65 characters)")
ax.set_ylabel("Classification accuracy"); ax.set_ylim(0, 1.05)
ax.grid(alpha=0.4, linestyle="--"); ax.legend(loc="lower left")
letter(ax, "c)", x=-0.2)

# d) per-participant UCIM, DistilBERT vs DeepSeek
ax = fig.add_subplot(gs[2, :])
diff = pu.ds_yes - pu.bert_yes
sc = ax.scatter(pu.bert_yes, pu.ds_yes, c=diff, cmap="coolwarm", vmin=-0.6, vmax=0.6, s=110,
                edgecolor="black", linewidth=1.2, zorder=3)
ax.plot([0, 1.05], [0, 1.05], "--", color=DS, lw=2, alpha=0.8, label="equal UCIM (y = x)", zorder=1)
for (xb, yd), grp in pu.groupby(["bert_yes", "ds_yes"]):
    ax.annotate("p" + ",".join(str(i) for i in grp.index), (xb, yd), xytext=(7, 5),
                textcoords="offset points", fontsize=11)
cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("UCIM difference (DeepSeek − DistilBERT)", fontsize=11)
ax.set_xlim(0.25, 1.05); ax.set_ylim(0.25, 1.05)
ax.set_xlabel("DistilBERT UCIM"); ax.set_ylabel("DeepSeek UCIM")
ax.grid(alpha=0.3); ax.legend(loc="lower right")
title(ax, "Per-participant user-confirmed intent match (participants p1–p20)")
letter(ax, "d)", x=-0.07)
save(fig, "Fig_Exp2_combined_revised")

# ========================================================================= Experiment 2: per-class P/R/F1
# Pooled over all 200 commands per model. The original figure (PRF_Both_V1) averaged per-participant
# scores; with 10 commands per participant most classes have 0-2 examples per person and absent classes
# were scored 0, which biased the bars downwards.
from sklearn.metrics import precision_recall_fscore_support
PRF_ORDER = ["grasp", "loosen", "tight_grasp", "open_hand", "pinch_grasp", "gentle_grasp", "tighten"]
cols = {"DistilBERT": ["#4a7ebb", "#7fa7e9", "#3a5f9e"], "DeepSeek": ["#ff9900", "#ffc966", "#cc7a00"]}
fig, ax = plt.subplots(figsize=(13, 4.8))
w = 0.8 / 6; x = np.arange(len(PRF_ORDER))
for m, (pred, gold) in enumerate((("bert_pred", "bert_gold"), ("ds_pred", "ds_gold"))):
    name = ["DistilBERT", "DeepSeek"][m]
    s = E2[E2[gold].notna()]
    pr, rc, f1, _ = precision_recall_fscore_support(s[gold], s[pred], labels=PRF_ORDER, zero_division=0)
    for j, (vals, lab) in enumerate(((pr, "Precision"), (rc, "Recall"), (f1, "F1"))):
        ax.bar(x - 0.4 + w / 2 + (3 * m + j) * w, vals, w, color=cols[name][j], label=f"{name} {lab}")
ax.set_xticks(x); ax.set_xticklabels([NICE[l] for l in PRF_ORDER])
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_xlabel("Class")
ax.grid(axis="y", alpha=0.4, linestyle="--")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
title(ax, "Per-class precision, recall and F1 for DistilBERT and DeepSeek (pooled over 200 commands each)")
save(fig, "Fig_Exp2_PRF_revised")

# ========================================================================= Experiment 4 (1)
S = pd.read_csv(OUT / "exp4_summary.csv")
P = list(S.participant)
fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.25)

ax = fig.add_subplot(gs[0, :])
x = np.arange(len(P)); w = 0.32
bb = ax.bar(x - w / 2, S.base, w, color=BERT_LIGHT, label="DistilBERT before adaptation")
bp = ax.bar(x + w / 2, S.pers, w, color=BERT_DARK, label="DistilBERT after adaptation (personalised)")
ax.scatter(x + w / 2 + 0.25, S.deepseek, marker="D", s=60, color=DS, edgecolor="black", zorder=4,
           label="DeepSeek zero-shot (reference)")
for i in range(len(P)):
    ax.text(x[i] - w / 2, S.base[i] + 0.015, f"{S.base[i]:.2f}", ha="center", fontsize=9, color="#1F497D", fontweight="bold")
    ax.text(x[i] + w / 2, S.pers[i] + 0.015, f"{S.pers[i]:.2f}", ha="center", fontsize=9, color="#1F2F5A", fontweight="bold")
    ax.text(x[i] + w / 2 + 0.25, S.deepseek[i] + 0.035, f"{S.deepseek[i]:.2f}", ha="center", fontsize=8, color="#9a5c00")
    ax.annotate("", xy=(x[i] + w / 2, S.pers[i]), xytext=(x[i] - w / 2, S.base[i]),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.4))
    ax.text(x[i], max(S.pers[i], S.base[i]) + 0.14, f"{S.gain_pp[i]:+.1f}", ha="center", fontsize=9,
            color="green", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
ax.axhline(S.base.mean(), color=BERT_LIGHT, ls=":", lw=1.3)
ax.axhline(S.pers.mean(), color=BERT_DARK, ls=":", lw=1.3)
ax.set_xticks(x); ax.set_xticklabels(list(P))
ax.set_ylim(0, 1.3); ax.set_ylabel("Held-out accuracy (70 prompts)"); ax.set_xlabel("Participants")
ax.legend(loc="upper center", ncol=3, fontsize=8.5, frameon=True); ax.grid(axis="y", alpha=0.3)
title(ax, "Accuracy on the held-out prompts before and after adaptation (gain in percentage points)")
letter(ax, "a)", x=-0.06)

ax = fig.add_subplot(gs[1, 0])
upd = []
for u in PARTICIPANTS:
    run = sorted((STUDY_DIR / "experiment_results" / u).glob("*/results_training_phase.csv"))[-1]
    T = pd.read_csv(run)
    upd.append((PAPER_IDS[u], T))
counts = [int(T.update_num.notna().sum()) for _, T in upd]
bars = ax.bar([k for k, _ in upd], counts, color=[P_COL[k] for k, _ in upd], width=0.55)
for b_, c in zip(bars, counts):
    ax.text(b_.get_x() + b_.get_width() / 2, c + 1.5, str(c), ha="center", fontweight="bold")
ax.axhline(np.mean(counts), color="black", ls="--", lw=1.2, label=f"Mean = {np.mean(counts):.1f}")
ax.set_ylim(0, 100); ax.set_ylabel("Model updates (of 140 prompts)"); ax.set_xlabel("Participant")
ax.legend(); ax.grid(axis="y", alpha=0.3)
title(ax, "Corrections that triggered an update")
letter(ax, "b)", x=-0.14)

ax = fig.add_subplot(gs[1, 1])
for k, T in upd:
    L = T.dropna(subset=["loss"]).reset_index(drop=True)
    xs = np.arange(1, len(L) + 1)
    ax.plot(xs, L.loss, "-", color=P_COL[k], lw=1, alpha=0.55)
    ax.plot(xs, L.loss.rolling(5, min_periods=1).mean(), "--", color=P_COL[k], lw=2, label=k)
ax.set_xlabel("Update number"); ax.set_ylabel("Cross-entropy loss")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
title(ax, "Loss on the replay mini-batch (dashed: rolling mean of 5)")
letter(ax, "c)", x=-0.14)
save(fig, "Fig_Exp4_personalisation_revised")

# ========================================================================= Experiment 4 (2)
F = pd.read_csv(OUT / "exp4_perclass_f1.csv")
I = pd.read_csv(OUT / "exp4_items.csv")
C = pd.read_csv(OUT / "exp4_cross.csv", index_col=0)
order = ["grasp", "open_hand", "gentle_grasp", "tight_grasp", "pinch_grasp", "tighten", "loosen"]
fig = plt.figure(figsize=(13, 9.5))
gs = gridspec.GridSpec(2, 2, height_ratios=[0.8, 1.05], hspace=0.4, wspace=0.3)

ax = fig.add_subplot(gs[0, :])
g = F.groupby("label")[["f1_base", "f1_pers"]].agg(["mean", "std"]).reindex(order)
x = np.arange(len(order)); w = 0.36
ax.bar(x - w / 2, g[("f1_base", "mean")], w, yerr=g[("f1_base", "std")], capsize=4, color=BERT_LIGHT,
       label="Before adaptation", error_kw=dict(ecolor="grey", elinewidth=1.1))
ax.bar(x + w / 2, g[("f1_pers", "mean")], w, yerr=g[("f1_pers", "std")], capsize=4, color=BERT_DARK,
       label="After adaptation (personalised)", error_kw=dict(ecolor="grey", elinewidth=1.1))
ax.set_xticks(x); ax.set_xticklabels([NICE[l].title() for l in order])
ax.set_ylim(0, 1.15); ax.set_ylabel("F1 score (mean ± SD)"); ax.set_xlabel("Grasp type")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
title(ax, "Per-class F1 of DistilBERT on the held-out prompts (4 participants)")
letter(ax, "a)", x=-0.09)

ax = fig.add_subplot(gs[1, 0])
cm = pd.crosstab(pd.Categorical(I.label, order), pd.Categorical(I.pers_pred, order), dropna=False)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, square=True, cbar_kws={"label": "Predictions (40 per class)"},
            xticklabels=[NICE[l].title() for l in order], yticklabels=[NICE[l].title() for l in order])
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.tick_params(axis="x", rotation=45)
title(ax, "Personalised models, pooled (4 participants, n = 280)")
letter(ax, "b)", x=-0.42)

ax = fig.add_subplot(gs[1, 1])
sns.heatmap(C, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, square=True,
            linewidths=1, linecolor="white", cbar_kws={"label": "Accuracy"})
for i in range(len(C)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2.5))
ax.set_xlabel("Held-out prompts of participant"); ax.set_ylabel("Model personalised to participant")
title(ax, "Cross-participant accuracy (held-out prompts only)")
letter(ax, "c)", x=-0.25)
save(fig, "Fig_Exp4_classes_cross_revised")

(FIG / "README.txt").write_text("""What changed in the Results figures, and why
=============================================
Fig_Exp2_combined_revised (replaces Combined_Giga_Figure_V4.pdf)
  a) unchanged content: pooled confusion matrices, DistilBERT blue, DeepSeek orange.
  b) same data; y-axis renamed from 'Accuracy' to 'User-confirmed intent match' (it is the share
     of commands each participant confirmed), as required by R2 #2/#8 and R3 #6.
  c) 'bins' (default): accuracy in four length quartiles over ALL 200 commands, with 95% Wilson
     intervals. The original panel dropped commands longer than 65 characters and fitted a
     quadratic whose band exceeded 1.0. Run with --length-panel fit to reproduce the original.
  d) same data; 'task completion success rate' renamed to UCIM (physical task success was not
     measured), as required by R2 #2/#4/#8.
Fig_Exp2_PRF_revised (replaces PRF_Both_V1.pdf)
  Same layout and colours. Scores are now pooled over all 200 commands per model. The original averaged
  per-participant scores, although most participants have only 0-2 commands of a class and classes a
  participant never used were scored 0, which pulled the bars down (e.g. DeepSeek 'loosen' F1 0.40 vs 0.71).
Fig_Exp4_personalisation_revised (replaces P_study1.pdf)
  a) 'Baseline' is now the frozen base model on the SAME 70 held-out prompts. The original used
     the running accuracy during the 140 adaptation prompts (while the model was already
     learning). DeepSeek zero-shot on the same prompts added as an orange reference (R1.1).
     Colours: lighter blue = DistilBERT before, darker blue = DistilBERT after, orange = DeepSeek.
  b) unchanged.  The former scatter (corrections vs gain) is removed: with the corrected
     baseline there is no positive relation (Spearman rho = -0.21, n = 4).
  c) loss curves, unchanged.
Fig_Exp4_classes_cross_revised (replaces P_study2.pdf)
  a) per-class F1 with the corrected (frozen-model) baseline.
  b) unchanged.
  c) cross-participant matrix computed on the held-out prompts only, with the saved personalised
     models. The original diagonal (0.78-0.85) does not match the personalised held-out accuracies
     (0.59-0.80), so it was not computed on the held-out prompts alone.
""", encoding="utf-8")
print("written", FIG / "README.txt")
