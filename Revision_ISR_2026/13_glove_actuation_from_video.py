# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
13_glove_actuation_from_video.py - approximate actuation time of the glove from existing videos.

Source: "Latency Videos/Glove_Very_Low.mp4" (same glove, recorded for an earlier study; 50 fps,
four camera views). An on-screen label shows when actuation is switched ON and when the
operator's hand pump is used ("Operator input: ON").

Method
  1. The label state is read in every frame (red "ON" text) -> actuation and hand-pump periods.
  2. Hand-pump activations are excluded (they are much slower and not driven by the regulators).
  3. For each actuation-ON event, the frame-to-frame image difference is computed in each camera
     view. Movement onset = first frame after the command where the (3-frame smoothed) difference
     exceeds the pre-command baseline mean + max(6 SD, 0.15); movement end = last frame above the
     threshold before 0.3 s of no movement.
  4. A check strip (every 2nd frame) is saved for every event so the result can be verified by eye.
Events in which the hand was already in the commanded posture produce no movement and are reported
as such. The other videos (YouTube, P4, nlpGlove demo) have no command marker and the fingers are
largely hidden, so they were not used for timing.

Also computes the serial transmission time of one command frame at 115200 baud.

Writes outputs/latency_video/actuation_from_video.md and strips/event_<frame>.png. Read-only otherwise.
Run:  python 13_glove_actuation_from_video.py
"""
from pathlib import Path

import cv2
import numpy as np

from common import OUT, ROOT

VIDEO = ROOT / "Voice_to_Grasp_Springer_revision" / "Latency Videos" / "Glove_Very_Low.mp4"
D = OUT / "latency_video"; (D / "strips").mkdir(parents=True, exist_ok=True)
VIEWS = {"top-left": (45, 240, 0, 355), "top-right": (45, 240, 360, 700),
         "bottom-left": (245, 480, 0, 355), "bottom-right": (245, 480, 360, 700)}

cap = cv2.VideoCapture(str(VIDEO)); fps = cap.get(cv2.CAP_PROP_FPS)
red = lambda r: float(((r[..., 2] > 150) & (r[..., 1] < 90) & (r[..., 0] < 90)).mean())
act, pump, M, prev = [], [], {k: [] for k in VIEWS}, None
while True:
    ok, f = cap.read()
    if not ok:
        break
    act.append(red(f[5:42, 100:165]) > 0.05)
    pump.append(red(f[5:42, 360:425]) > 0.05)
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    for k, (y0, y1, x0, x1) in VIEWS.items():
        M[k].append(0.0 if prev is None else float(np.abs(g[y0:y1, x0:x1] - prev[y0:y1, x0:x1]).mean()))
    prev = g
act, pump = np.array(act), np.array(pump)
M = {k: np.convolve(np.array(v), np.ones(3) / 3, "same") for k, v in M.items()}
on_events = [int(i) + 1 for i in np.flatnonzero(np.diff(act.astype(int)) == 1)]
pump_periods = [(int(a) + 1, int(b) + 1) for a, b in zip(np.flatnonzero(np.diff(pump.astype(int)) == 1),
                                                        np.flatnonzero(np.diff(pump.astype(int)) == -1))]


def strip(f0, f1, tag):
    c = cv2.VideoCapture(str(VIDEO)); tiles = []
    for fr in range(f0, f1, 2):
        c.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, im = c.read()
        if not ok:
            break
        im = cv2.resize(im, (240, 160))
        cv2.putText(im, f"{fr} ({(fr - f0) / fps:.2f}s)", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        tiles.append(im)
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(D / "strips" / f"{tag}.png"), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))


lines = ["# Glove actuation time from video (13_glove_actuation_from_video.py)\n",
         f"Video: {VIDEO.name}, {fps:.0f} fps; actuation-ON events at frames {on_events}; "
         f"hand-pump periods (excluded) at frames {pump_periods}\n",
         "| event (frame) | view | onset after command (ms) | movement duration (ms) | command to end of movement (ms) |",
         "|---|---|---|---|---|"]
valid = []
for ev in on_events:
    strip(ev - 10, ev + 120, f"event_{ev}")
    for k, m in M.items():
        base = m[ev - 50:ev - 2]; thr = base.mean() + max(6 * base.std(), 0.15)
        w = m[ev:ev + 250]; above = np.flatnonzero(w > thr)
        if len(above) == 0 or above[0] / fps > 1.0:   # no movement within 1 s: hand already in posture
            lines.append(f"| {ev} | {k} | no movement | - | - |"); continue
        on = int(above[0]); end = on
        for i in range(on, len(w)):
            if w[i] > thr:
                end = i
            elif i - end >= int(0.3 * fps):
                break
        row = (1000 * on / fps, 1000 * (end - on) / fps, 1000 * end / fps)
        lines.append(f"| {ev} | {k} | {row[0]:.0f} | {row[1]:.0f} | {row[2]:.0f} |")
        if row[1] >= 500:      # a complete finger movement (shorter detections are partial)
            valid.append(row)
if valid:
    v = np.array(valid)
    lines.append(f"\nComplete movements: n = {len(v)} (camera views of the same activation(s)); onset "
                 f"{v[:, 0].min():.0f}-{v[:, 0].max():.0f} ms after the command; movement {v[:, 1].min():.0f}-{v[:, 1].max():.0f} ms; "
                 f"command to grasp posture {v[:, 2].min():.0f}-{v[:, 2].max():.0f} ms.")
frame = "45 45\n".encode()
lines.append(f"\nSerial transmission of one command frame ({len(frame)} bytes, 10 bits per byte at 115200 baud): "
             f"{1000 * len(frame) * 10 / 115200:.2f} ms.")
(D / "actuation_from_video.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
