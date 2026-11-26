# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
cl_core.py - a configurable, in-memory re-implementation of the Voice-to-Grasp
continual-learning (update-on-error + experience replay) loop.

Default config "paper" reproduces Continual_Learning_Experiments/
UserStudy_code_analysis/continual_trainer.py exactly as it was run for Exp 4:
  * every sample (correct or not) is added to the buffer
  * buffer: 100 slots, once full a RANDOM slot is overwritten ("random_replace")
  * on error: one AdamW step on a random mini-batch of <=25 buffer samples
    (the current sample is only included if it happens to be drawn)
  * dual LR: head 9e-4, DistilBERT backbone 1e-5, grad-clip 1.0

Everything else is a switch, so the reviewers' questions can be answered by
ablation (learning-rate strategy, LoRA / frozen backbone = structural
separation, buffer policy and size, anchor replay, noisy corrections, gating).

Nothing is written to disk; the original adaptive_model folders are untouched.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, LABEL2ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CLConfig:
    name: str = "paper"
    update: str = "dual"            # dual | single | head_only | lora | none
    head_lr: float = 9e-4
    backbone_lr: float = 1e-5       # used by dual / single (single uses head_lr for all)
    lora_r: int = 8
    lora_lr: float = 5e-4
    buffer_policy: str = "random_replace"   # fifo | random_replace | reservoir | class_balanced | none
    buffer_size: int = 100
    replay_size: int = 25
    add_correct: bool = True        # add correctly-predicted samples to the buffer
    include_current: bool = False   # force the misclassified sample into the batch
    anchor_k: int = 0               # base-training samples kept permanently in the replay pool
    noise: float = 0.0              # prob. that a user correction carries a wrong label
    loss_gate: float = 0.0          # >0: skip update if CE of corrected label > gate (implausible correction)
    # The original code calls optimizer.zero_grad(), which does NOT clear the
    # gradients of pre_classifier (it is not in the optimizer). Those gradients
    # accumulate and inflate the global norm used by clip_grad_norm_, so the
    # effective step size shrinks with every update. fix_grad=True clears all
    # grads (model.zero_grad) and is used in the "fixed" configurations.
    fix_grad: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def as_row(self):
        d = asdict(self)
        d.pop("extra")
        return d


# --------------------------------------------------------------------------
# Model cache - load the base checkpoint once, deep-copy per run
# --------------------------------------------------------------------------
_CACHE = {}


def load_base(path):
    key = str(path)
    if key not in _CACHE:
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path)
        mdl.eval()
        _CACHE[key] = (tok, mdl)
    return _CACHE[key]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _backbone(model):
    for attr in ("distilbert", "bert", "roberta"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError("unknown backbone")


def _head_params(model):
    names = [n for n, _ in model.named_parameters()
             if n.startswith(("pre_classifier", "classifier"))]
    return [p for n, p in model.named_parameters() if n in names]


@torch.no_grad()
def predict(model, tok, texts, bs=64):
    """Return (pred_ids, probs) for a list of texts."""
    model.eval()
    preds, probs = [], []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), return_tensors="pt", truncation=True,
                  padding=True, max_length=128).to(DEVICE)
        p = torch.softmax(model(**enc).logits, dim=-1).cpu()
        probs.append(p)
        preds.append(p.argmax(-1))
    return torch.cat(preds).numpy(), torch.cat(probs).numpy()


@torch.no_grad()
def cls_embeddings(model, tok, texts, bs=64):
    """[CLS] hidden state of the backbone - used to measure representation drift."""
    model.eval()
    bb = _backbone(model)
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), return_tensors="pt", truncation=True,
                  padding=True, max_length=128).to(DEVICE)
        h = bb(**enc).last_hidden_state[:, 0, :]
        out.append(h.cpu())
    return torch.cat(out).numpy()


def linear_cka(x, y):
    """Linear CKA (Kornblith et al. 2019). 1.0 = identical representations."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    hsic = np.linalg.norm(x.T @ y, "fro") ** 2
    return float(hsic / (np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")))


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------
class Buffer:
    def __init__(self, policy, size, rng):
        self.policy, self.size, self.rng = policy, size, rng
        self.items = []                 # list of (text, label)
        self.seen = 0
        self.per_class = {l: [] for l in LABELS}
        self.seen_class = {l: 0 for l in LABELS}

    def add(self, text, label):
        self.seen += 1
        if self.policy == "none" or self.size <= 0:
            return
        if self.policy == "class_balanced":
            q = max(1, self.size // len(LABELS))
            bucket = self.per_class[label]
            self.seen_class[label] += 1
            if len(bucket) < q:
                bucket.append((text, label))
            else:                        # reservoir sampling within the class
                j = self.rng.randrange(self.seen_class[label])
                if j < q:
                    bucket[j] = (text, label)
            self.items = [x for b in self.per_class.values() for x in b]
            return
        if len(self.items) < self.size:
            self.items.append((text, label))
        elif self.policy == "fifo":
            self.items.pop(0)
            self.items.append((text, label))
        elif self.policy == "random_replace":       # what the Exp-4 code does
            self.items[self.rng.randrange(self.size)] = (text, label)
        elif self.policy == "reservoir":            # Vitter's algorithm R
            j = self.rng.randrange(self.seen)
            if j < self.size:
                self.items[j] = (text, label)
        else:
            raise ValueError(self.policy)

    def sample(self, k):
        return self.rng.sample(self.items, min(k, len(self.items)))


# --------------------------------------------------------------------------
# The learner
# --------------------------------------------------------------------------
class ContinualLearner:
    def __init__(self, cfg: CLConfig, base_path, anchor_pool=None):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        tok, base = load_base(base_path)
        self.tok = tok
        self.model = copy.deepcopy(base).to(DEVICE)
        self.buffer = Buffer(cfg.buffer_policy, cfg.buffer_size, self.rng)
        self.anchors = []
        if cfg.anchor_k and anchor_pool is not None and len(anchor_pool):
            idx = self.rng.sample(range(len(anchor_pool)), min(cfg.anchor_k, len(anchor_pool)))
            self.anchors = [(anchor_pool.iloc[i]["prompt"], anchor_pool.iloc[i]["label"]) for i in idx]
        self.n_updates = 0
        self.n_gated = 0
        self.losses = []
        self._build_optimizer()

    def _build_optimizer(self):
        c, m = self.cfg, self.model
        if c.update == "none":
            self.opt = None
            return
        if c.update == "lora":
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(r=c.lora_r, lora_alpha=2 * c.lora_r, lora_dropout=0.0,
                              target_modules=["q_lin", "v_lin"],
                              modules_to_save=["pre_classifier", "classifier"])
            self.model = get_peft_model(m, lcfg)
            lora = [p for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad]
            head = [p for n, p in self.model.named_parameters()
                    if p.requires_grad and "lora_" not in n]
            self.opt = torch.optim.AdamW([{"params": head, "lr": c.head_lr},
                                          {"params": lora, "lr": c.lora_lr}])
            return
        head = _head_params(m)
        head_ids = {id(p) for p in head}
        bb = [p for p in m.parameters() if id(p) not in head_ids]
        if c.update == "dual":
            # identical grouping to the original code: classifier @ head_lr,
            # distilbert backbone @ backbone_lr. NOTE: pre_classifier (768x768) is
            # NOT in the original optimizer, i.e. it is never updated. Set
            # extra={"train_pre_classifier": True} to include it at head_lr.
            groups = [{"params": list(m.classifier.parameters()), "lr": c.head_lr},
                      {"params": list(_backbone(m).parameters()), "lr": c.backbone_lr}]
            if c.extra.get("train_pre_classifier", False):
                groups.append({"params": list(m.pre_classifier.parameters()), "lr": c.head_lr})
        elif c.update == "single":
            groups = [{"params": list(m.parameters()), "lr": c.extra.get("lr", c.backbone_lr)}]
        elif c.update == "head_only":
            for p in bb:
                p.requires_grad_(False)
            groups = [{"params": head, "lr": c.head_lr}]
        else:
            raise ValueError(c.update)
        self.opt = torch.optim.AdamW(groups)

    # ------------------------------------------------------------------
    def predict_one(self, text):
        ids, probs = predict(self.model, self.tok, [text])
        return LABELS[int(ids[0])], float(probs[0].max()), probs[0]

    def observe(self, text, true_label, accepted=None):
        """One interaction: predict, then learn from the user's feedback.
        accepted: optional set of labels the user would accept (Exp 2 data).
        Returns (pred, correct)."""
        c = self.cfg
        pred, conf, probs = self.predict_one(text)
        correct = (pred in accepted) if accepted else (pred == true_label)
        label_given = true_label
        if not correct and c.noise > 0 and self.rng.random() < c.noise:
            label_given = self.rng.choice([l for l in LABELS if l not in (true_label,)])
        if correct and c.add_correct:
            self.buffer.add(text, pred if accepted else true_label)
        if not correct:
            self.buffer.add(text, label_given)
            if c.update != "none":
                if c.loss_gate > 0 and -np.log(max(probs[LABEL2ID[label_given]], 1e-12)) > c.loss_gate:
                    self.n_gated += 1
                else:
                    self._update(text, label_given)
        return pred, correct

    def _update(self, text, label):
        c = self.cfg
        batch = self.buffer.sample(c.replay_size)
        if c.include_current and (text, label) not in batch:
            batch = batch + [(text, label)]
        if self.anchors:
            batch = batch + self.rng.sample(self.anchors, min(len(self.anchors), max(1, c.replay_size // 5)))
        if not batch:
            batch = [(text, label)]
        self.model.train()
        enc = self.tok([t for t, _ in batch], return_tensors="pt", truncation=True,
                       padding=True, max_length=128).to(DEVICE)
        y = torch.tensor([LABEL2ID[l] for _, l in batch], device=DEVICE)
        loss = cross_entropy(self.model(**enc).logits, y)
        if c.fix_grad:
            self.model.zero_grad(set_to_none=True)
        else:
            self.opt.zero_grad()        # faithful to the original code
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.opt.step()
        decay = c.extra.get("lr_decay")          # explicit per-update multiplicative LR decay
        if decay:
            for gr in self.opt.param_groups:
                gr["lr"] *= decay
        self.model.eval()
        self.n_updates += 1
        self.losses.append(float(loss.item()))

    def evaluate(self, texts, labels):
        """labels: list of str or list of set(str). Returns (acc, preds)."""
        ids, _ = predict(self.model, self.tok, list(texts))
        preds = [LABELS[i] for i in ids]
        ok = [(p in l) if isinstance(l, (set, frozenset)) else (p == l) for p, l in zip(preds, labels)]
        return float(np.mean(ok)), preds

    def embeddings(self, texts):
        m = self.model
        if hasattr(m, "get_base_model"):
            # merge LoRA weights into a copy to get the effective backbone
            m = copy.deepcopy(m).merge_and_unload()
        return cls_embeddings(m, self.tok, list(texts))
