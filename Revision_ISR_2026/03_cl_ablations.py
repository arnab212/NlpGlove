# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
03_cl_ablations.py - ablation study for Reviewer 1 (and R3 #2).

Suites
  lr      : how should the update be applied?  (R1 "dual learning rate" concern)
            dual (paper) | dual+fixes | single LR 1e-5 | single LR 5e-5 |
            head-only (frozen backbone) | LoRA adapters (structural separation)
  buffer  : how should experience be kept?     (R1 "fixed 100-sample buffer" concern)
            no replay | FIFO | random-replace (paper) | reservoir | class-balanced
            x sizes 25/50/100/140, + anchor replay of base-training data
  noise   : robustness to wrong user corrections (R1 "malfunction feedback")
            noise 0/10/20/30 % x {dual, head-only, LoRA} (+ loss-gate variant)
  all     : everything above

Every run: participants x seeds, base DistilBERT re-loaded fresh each time.
Results are appended to outputs/ablations_<suite>.csv (resumable: finished
(config, user, seed) triples are skipped).

Usage (from the Revision_ISR_2026 folder, inside the nlpglove venv):
    python 03_cl_ablations.py --suite lr --seeds 0 1 2 3 4
    python 03_cl_ablations.py --suite all --seeds 0 1 2 --users P1 P2 P3 P4
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace

from cl_core import CLConfig
from common import OUT, PARTICIPANTS
from runner import run_one

PAPER = CLConfig(name="paper")
# "fixed" = same method with the three code issues corrected:
#   current sample always in the batch, all grads zeroed, pre_classifier trained
FIXED = replace(PAPER, name="dual_fixed", include_current=True, fix_grad=True,
                extra={"train_pre_classifier": True})


def suite_lr():
    return [
        PAPER,
        FIXED,
        # isolate the three code differences one at a time
        replace(PAPER, name="paper+current_in_batch", include_current=True),
        replace(PAPER, name="paper+zero_all_grads", fix_grad=True),
        replace(PAPER, name="paper+train_pre_classifier", extra={"train_pre_classifier": True}),
        # the accumulated-gradient effect acts like a decaying step size; make it explicit
        replace(PAPER, name="dual_explicit_lr_decay", fix_grad=True, include_current=True,
                extra={"lr_decay": 0.97}),
        replace(FIXED, name="single_lr_1e-5", update="single", extra={"lr": 1e-5}),
        replace(FIXED, name="single_lr_5e-5", update="single", extra={"lr": 5e-5}),
        replace(FIXED, name="single_lr_9e-4", update="single", extra={"lr": 9e-4}),
        replace(FIXED, name="head_only", update="head_only"),
        replace(PAPER, name="classifier_only", update="dual", backbone_lr=0.0),
        replace(FIXED, name="lora_r8", update="lora", lora_r=8),
        replace(FIXED, name="no_adaptation", update="none"),
    ]


def suite_buffer():
    # built on the paper configuration (best in the lr suite)
    cfgs = [replace(PAPER, name="replay_none", buffer_policy="none", replay_size=0)]
    for pol in ("fifo", "random_replace", "reservoir", "class_balanced"):
        for size in (25, 50, 100, 140):
            cfgs.append(replace(PAPER, name=f"buf_{pol}_{size}", buffer_policy=pol, buffer_size=size))
    cfgs.append(replace(PAPER, name="buf_class_balanced_100+anchor50",
                        buffer_policy="class_balanced", anchor_k=50))
    return cfgs


def suite_noise():
    cfgs = []
    for nz in (0.0, 0.1, 0.2, 0.3):
        tag = int(nz * 100)
        cfgs += [
            replace(PAPER, name=f"noise{tag}_dual", noise=nz),
            replace(PAPER, name=f"noise{tag}_classifier_only", backbone_lr=0.0, noise=nz),
            replace(FIXED, name=f"noise{tag}_lora", update="lora", noise=nz),
            replace(PAPER, name=f"noise{tag}_dual_gate", noise=nz, loss_gate=4.0),
        ]
    return cfgs


SUITES = {"lr": suite_lr, "buffer": suite_buffer, "noise": suite_noise}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="lr", choices=list(SUITES) + ["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--users", nargs="+", default=list(PARTICIPANTS))
    ap.add_argument("--only", nargs="*", default=None, help="run only these config names")
    args = ap.parse_args()

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    for s in suites:
        cfgs = SUITES[s]()
        if args.only:
            cfgs = [c for c in cfgs if c.name in args.only]
        out = OUT / f"ablations_{s}.csv"
        done = set()
        if out.exists():
            with open(out, newline="") as f:
                for r in csv.DictReader(f):
                    done.add((r["name"], r["user"], int(r["seed"])))
        header_written = out.exists() and os.path.getsize(out) > 0
        total = len(cfgs) * len(args.users) * len(args.seeds)
        k = 0
        for cfg in cfgs:
            for seed in args.seeds:
                for user in args.users:
                    k += 1
                    if (cfg.name, user, seed) in done:
                        continue
                    row = run_one(replace(cfg, seed=seed), user,
                                  cross_users=(s == "lr"), drift=True)
                    if header_written:
                        with open(out, newline="") as f:
                            fieldnames = next(csv.reader(f))
                    else:
                        fieldnames = list(row.keys())
                    with open(out, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                        if not header_written:
                            w.writeheader()
                            header_written = True
                        w.writerow(row)
                    print(f"[{s} {k}/{total}] {cfg.name:32s} {user:7s} seed={seed} "
                          f"base={row['base_test_acc']:.3f} -> {row['test_acc']:.3f} "
                          f"upd={row['n_updates']:3d} forget={row['forget_basedata_pp']:+.1f}pp "
                          f"cka={row.get('cka_backbone_vs_base', float('nan')):.3f} "
                          f"({row['seconds']:.0f}s)", flush=True)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
