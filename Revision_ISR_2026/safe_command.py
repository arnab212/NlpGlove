# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
safe_command.py - the output-validation / safety layer Reviewer 3 (#7) asks
about, written so it can be described in the paper and dropped into the GUI.

Pipeline:  model output  ->  validate_label()  ->  SafetyGate.decide()  ->  pressures

1. Whitelist: only the 7 labels are executable; anything else -> NO_ACTION and
   the user is asked to repeat (the original GUI silently did nothing, which is
   safe but undocumented).
2. Confidence gate (DistilBERT softmax / LLM self-reported score): below
   `min_conf` the system asks for confirmation instead of actuating.
3. State rules: tighten/loosen only act if a grasp is active; tighten can
   never exceed the pressure ceiling; open_hand is always allowed (it is the
   pain-relief / emergency command - scenario 10).
4. Hard limits: per-channel pressure clamped to [0, P_MAX_PCT]; maximum rate of
   change per command; independent of the model.

PRESSURES below are the values the glove ACTUALLY executed in GuiUserStudyNew.py
(percent of the regulator range; 100 % of the 0-10 V command = regulator full
scale, NOT the 69 kPa nominal - check and state the kPa equivalent in the paper).
"""
from __future__ import annotations

from dataclasses import dataclass

LABELS = ["grasp", "open_hand", "gentle_grasp", "tight_grasp",
          "pinch_grasp", "tighten", "loosen"]

# (channel-1 %, channel-2 %) as executed in the user study GUI
PRESSURES = {
    "open_hand":    (0, 0),
    "gentle_grasp": (30, 30),
    "grasp":        (45, 45),
    "pinch_grasp":  (50, 0),      # channel 1 only -> confirm which fingers channel 1 drives
    "tight_grasp":  (70, 70),
}
STEP = 25                          # tighten / loosen step (% points) used in the study
P_MAX_PCT = 95                     # ceiling for tighten (original GUI allowed up to 100 %)


def validate_label(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip().lower().replace(" ", "_").replace("-", "_")
    return s if s in LABELS else None


@dataclass
class Decision:
    action: str            # EXECUTE | ASK_CONFIRM | ASK_REPEAT | NO_CHANGE
    label: str | None
    pressures: tuple | None
    reason: str


class SafetyGate:
    def __init__(self, min_conf=0.5, p_max=P_MAX_PCT, step=STEP):
        self.min_conf, self.p_max, self.step = min_conf, p_max, step
        self.state = (0, 0)

    def decide(self, raw_label, confidence=None) -> Decision:
        lab = validate_label(raw_label)
        if lab is None:
            return Decision("ASK_REPEAT", None, None, f"output '{raw_label}' is not an executable label")
        if lab == "open_hand":                       # always allowed, never gated
            return self._go(lab, (0, 0), "release is always permitted")
        if confidence is not None and confidence < self.min_conf:
            return Decision("ASK_CONFIRM", lab, None, f"confidence {confidence:.2f} < {self.min_conf}")
        if lab in ("tighten", "loosen"):
            if self.state == (0, 0):
                return Decision("NO_CHANGE", lab, self.state, f"'{lab}' ignored: no active grasp")
            sgn = 1 if lab == "tighten" else -1
            new = tuple(0 if c == 0 else max(0, min(self.p_max, c + sgn * self.step)) for c in self.state)
            if new == self.state:
                return Decision("NO_CHANGE", lab, self.state, "already at limit")
            return self._go(lab, new, "relative adjustment within limits")
        return self._go(lab, PRESSURES[lab], "absolute grasp level")

    def _go(self, lab, p, why):
        p = tuple(max(0, min(self.p_max, v)) for v in p)
        self.state = p
        return Decision("EXECUTE", lab, p, why)


if __name__ == "__main__":       # tiny self-test
    g = SafetyGate()
    for lab, conf in [("tighten", .9), ("grasp", .9), ("tighten", .9), ("tighten", .9),
                      ("hold_it", .9), ("tight_grasp", .3), ("loosen", .8), ("open_hand", .1)]:
        d = g.decide(lab, conf)
        print(f"{lab:12s} conf={conf:.1f} -> {d.action:11s} {d.pressures}  ({d.reason})")
