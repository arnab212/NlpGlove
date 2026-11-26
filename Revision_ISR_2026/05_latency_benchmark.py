# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
05_latency_benchmark.py - end-to-end timing + hardware/software inventory
(R2 #1, #9; R3 #3). Run on the SAME lab PC used in the study.

Stages measured (each N times, median / IQR / p95 reported):
  0. env       : CPU, GPU, RAM, OS, Python/torch/transformers/whisper versions
  1. capture   : audio window length (the GUI records a fixed 7 s window)
  2. asr       : Whisper 'base' transcription time per utterance
                 (--audio a.wav b.mp3 ...  or  --record N to speak N commands)
  3. distilbert: tokenisation + forward pass, CPU and (if present) GPU,
                 on the 200 real Exp-2 transcripts; plus model load time
  4. deepseek  : API round trip for deepseek-reasoner (used in the study) and
                 deepseek-chat, token usage -> cost per command
                 (needs DEEPSEEK_API_KEY; --deepseek-n 20)
  5. local_llm : optional, --gguf path  (llama-cpp-python)
  6. serial    : PC -> Arduino -> PC echo round trip (Arduino prints a line
                 after every command), --port COM5
  7. actuation : optional --actuation: sends a grasp, you press ENTER when the
                 glove has visibly finished moving (coarse, +/-0.2 s). For a
                 publishable number film the glove at >=120 fps together with
                 the screen (script flashes the console) and count frames.

Writes outputs/latency_<timestamp>.csv and outputs/latency_report.md

Examples
  python 05_latency_benchmark.py --record 10 --port COM5
  python 05_latency_benchmark.py --audio ..\\gentle.mp3 ..\\tight.mp3 --deepseek-n 20
  python 05_latency_benchmark.py --skip-asr --port COM5 --actuation
"""
from __future__ import annotations

import argparse
import platform
import statistics as st
import time
from datetime import datetime

import numpy as np
import pandas as pd

from common import BASE_MODEL, OUT, load_exp2

ROWS = []


def rec(stage, seconds, **kw):
    ROWS.append({"stage": stage, "seconds": seconds, **kw})


def summary(stage):
    v = [r["seconds"] for r in ROWS if r["stage"] == stage]
    if not v:
        return None
    a = np.array(v)
    return {"stage": stage, "n": len(a), "median_ms": 1000 * np.median(a),
            "iqr_ms": f"{1000*np.percentile(a,25):.0f}-{1000*np.percentile(a,75):.0f}",
            "p95_ms": 1000 * np.percentile(a, 95), "max_ms": 1000 * a.max()}


def env_info():
    info = {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu_cores"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    for mod in ("transformers", "whisper", "sounddevice", "serial", "openai"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = "not installed"
    return info


def bench_asr(args):
    import whisper
    t0 = time.perf_counter()
    model = whisper.load_model(args.whisper)
    rec("asr_model_load", time.perf_counter() - t0)
    clips = []
    if args.record:
        import sounddevice as sd
        sr = 16000
        for i in range(args.record):
            input(f"[{i+1}/{args.record}] press ENTER, then speak a grasp command "
                  f"(recording {args.window:.0f} s)...")
            audio = sd.rec(int(args.window * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            clips.append(audio.flatten())
    for path in args.audio or []:
        clips.append(whisper.load_audio(path))      # needs ffmpeg on PATH for mp3
    for i, a in enumerate(clips):
        t0 = time.perf_counter()
        r = model.transcribe(a, fp16=False, language="en")
        rec("asr_whisper_" + args.whisper, time.perf_counter() - t0, text=r.get("text", "").strip(),
            audio_s=len(a) / 16000)
        print(f"  asr {i+1}: {ROWS[-1]['seconds']*1000:.0f} ms  '{ROWS[-1]['text']}'")


def bench_distilbert(texts):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    m = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).eval()
    rec("distilbert_load", time.perf_counter() - t0)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        m.to(dev)
        with torch.no_grad():
            for t in texts[:5]:                       # warm-up
                m(**tok(t, return_tensors="pt").to(dev))
            for t in texts:
                t0 = time.perf_counter()
                enc = tok(t, return_tensors="pt", truncation=True).to(dev)
                _ = m(**enc).logits.argmax(-1).item()
                if dev == "cuda":
                    torch.cuda.synchronize()
                rec(f"distilbert_{dev}", time.perf_counter() - t0)
        # one continual-learning update (25-sample batch) - the cost of "learning from a correction"
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
        for _ in range(5):
            batch = texts[:25]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(dev)
            y = torch.zeros(len(batch), dtype=torch.long, device=dev)
            t0 = time.perf_counter()
            loss = torch.nn.functional.cross_entropy(m(**enc).logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            rec(f"cl_update_step_{dev}", time.perf_counter() - t0)
        m.eval()


def bench_llm(backend, texts, stage):
    for t in texts:
        try:
            r = backend(t)
            rec(stage, r["latency_s"], label=r["label"], prompt_tokens=r.get("prompt_tokens"),
                completion_tokens=r.get("completion_tokens"))
            print(f"  {stage}: {r['latency_s']*1000:.0f} ms -> {r['label']}")
        except Exception as ex:
            print(f"  {stage} failed: {ex}")


def bench_serial(args):
    import serial
    s = serial.Serial(args.port, 115200, timeout=2)
    time.sleep(2.5)                                   # Arduino resets on open
    s.reset_input_buffer()
    for i in range(args.serial_n):
        cmd = f"{(i % 2) * 30} {(i % 2) * 30}\n"      # alternate 0/30 %: low, safe pressure
        t0 = time.perf_counter()
        s.write(cmd.encode()); s.flush()
        line = s.readline()
        dt = time.perf_counter() - t0
        if line:
            rec("serial_roundtrip", dt)
        time.sleep(0.3)
    s.write(b"0 0\n")
    if args.actuation:
        print("\nACTUATION: watch the glove. Press ENTER when it has finished closing / opening.")
        for i in range(args.actuation_n):
            for name, cmd in (("close_grasp_45", b"45 45\n"), ("open_hand", b"0 0\n")):
                input(f"  [{i+1}] ready for {name}? ENTER to send...")
                print("\n" * 3 + "#" * 60 + f"\n   SENT {name}\n" + "#" * 60)
                t0 = time.perf_counter()
                s.write(cmd); s.flush()
                input("  ...ENTER when motion complete")
                rec(f"actuation_manual_{name}", time.perf_counter() - t0)
    s.write(b"0 0\n")
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="*")
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--window", type=float, default=7.0, help="capture window used by the GUI (s)")
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--deepseek-n", type=int, default=0)
    ap.add_argument("--deepseek-models", nargs="+", default=["deepseek-reasoner", "deepseek-chat"])
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (from DeepSeek pricing page)")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--serial-n", type=int, default=50)
    ap.add_argument("--actuation", action="store_true")
    ap.add_argument("--actuation-n", type=int, default=5)
    args = ap.parse_args()

    info = env_info()
    print("ENV:", info)
    texts = list(load_exp2().prompt)

    rec("capture_window_fixed", args.window)
    if not args.skip_asr and (args.audio or args.record):
        bench_asr(args)
    bench_distilbert(texts)
    if args.deepseek_n:
        from llm_backends import DeepSeekBackend
        for mname in args.deepseek_models:
            bench_llm(DeepSeekBackend(mname), texts[:args.deepseek_n], f"deepseek_{mname}")
    if args.gguf:
        from llm_backends import LlamaCppBackend
        bench_llm(LlamaCppBackend(args.gguf), texts[:20], "local_llm")
    if args.port:
        bench_serial(args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(ROWS)
    df.to_csv(OUT / f"latency_{stamp}.csv", index=False)
    S = [s for s in (summary(x) for x in df.stage.unique()) if s]
    md = [f"# Latency report ({stamp})\n", "## Environment (paste into the hardware/software table)"]
    md += [f"- {k}: {v}" for k, v in info.items()]
    md += ["\n## Stage timings", "| stage | n | median (ms) | IQR (ms) | p95 (ms) | max (ms) |", "|---|---|---|---|---|---|"]
    md += [f"| {s['stage']} | {s['n']} | {s['median_ms']:.0f} | {s['iqr_ms']} | {s['p95_ms']:.0f} | {s['max_ms']:.0f} |" for s in S]
    for mname in args.deepseek_models:
        g = df[df.stage == f"deepseek_{mname}"]
        if len(g) and args.price_in and args.price_out:
            cost = (g.prompt_tokens.mean() * args.price_in + g.completion_tokens.mean() * args.price_out) / 1e6
            md.append(f"\n{mname}: mean tokens in/out {g.prompt_tokens.mean():.0f}/{g.completion_tokens.mean():.0f} "
                      f"-> ~${cost:.5f} per command (~${cost*100:.3f} per 100 commands)")
    med = {s["stage"]: s["median_ms"] for s in S}
    def g(k):
        return med.get(k, float("nan"))
    asr = next((v for k, v in med.items() if k.startswith("asr_whisper")), float("nan"))
    md.append("\n## End-to-end (median) - fill the paper's latency table from this")
    md.append(f"- offline path : capture {args.window*1000:.0f} + ASR {asr:.0f} + DistilBERT {g('distilbert_cpu'):.0f} "
              f"+ serial/2 {g('serial_roundtrip')/2:.0f} + actuation {g('actuation_manual_close_grasp_45'):.0f} ms")
    md.append(f"- online path  : capture {args.window*1000:.0f} + ASR {asr:.0f} + DeepSeek-reasoner "
              f"{g('deepseek_deepseek-reasoner'):.0f} + serial/2 {g('serial_roundtrip')/2:.0f} + actuation "
              f"{g('actuation_manual_close_grasp_45'):.0f} ms")
    md.append("- Note the participant's transcript-confirmation click is a human step and is excluded; say so.")
    (OUT / "latency_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
