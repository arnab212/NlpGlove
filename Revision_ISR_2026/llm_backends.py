# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
llm_backends.py - one prompt, several back-ends, ALWAYS validated output.

  * DeepSeek API (deepseek-reasoner = what the user study used, deepseek-chat)
    -> API key is read from the environment variable DEEPSEEK_API_KEY.
       (The original GUI scripts have the key hard-coded - rotate it.)
  * Local GGUF model through llama-cpp-python (Llama-2 / Mistral files already
    in nlpglove/models, or any newer instruct model you download), with a GBNF
    grammar so the model can ONLY emit one of the 7 labels.

The prompt reproduces the one in GuiUserStudyNew.py, but the force levels
are corrected to the values the glove actually executed (see safe_command.py).
"""
from __future__ import annotations

import json
import os
import re
import time

from common import LABELS
from safe_command import validate_label

STRATEGIES = {
    "grasp": "Flex all fingers at medium force.",
    "open_hand": "Release / fully extend all fingers.",
    "gentle_grasp": "Flex all fingers at low force (fragile or light objects).",
    "tight_grasp": "Flex all fingers at high force (heavy objects).",
    "pinch_grasp": "Thumb-index precision grip (small or thin objects).",
    "tighten": "Increase the current grip force by one step (only if already grasping).",
    "loosen": "Decrease the current grip force by one step (only if already grasping).",
}


def build_messages(command_text: str):
    prompt_data = {
        "scenario": "The user is wearing a soft robotic hand exoskeleton and speaks a command.",
        "available_grasp_strategies": {str(i): {"name": n, "description": STRATEGIES[n]}
                                        for i, n in enumerate(LABELS)},
        "input": command_text,
        "output_format": {"strategy_id": "id, e.g. 2", "name": "strategy name, e.g. gentle_grasp",
                          "confidence_score": "0-1"},
    }
    return [
        {"role": "system", "content": "You are a grasp strategy selector for a soft robotic exoskeleton."},
        {"role": "user", "content": "Here is the command context and strategy mapping:\n"
                                    f"{json.dumps(prompt_data, indent=2)}\n"
                                    "Return the best grasp strategy. Respond only with a JSON object "
                                    "according to output_format."},
    ]


def parse_label(text: str):
    """Extract a label from model output; returns (label or None, raw)."""
    try:
        j = json.loads(text)
        lab = validate_label(j.get("name")) or (
            LABELS[int(j["strategy_id"])] if str(j.get("strategy_id", "")).isdigit()
            and int(j["strategy_id"]) < len(LABELS) else None)
        if lab:
            return lab
    except Exception:
        pass
    for l in sorted(LABELS, key=len, reverse=True):     # fall back: first label mentioned
        if re.search(rf"\b{l}\b", text):
            return l
    return None


class DeepSeekBackend:
    def __init__(self, model="deepseek-reasoner"):
        from openai import OpenAI
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("Set the environment variable DEEPSEEK_API_KEY (do not hard-code it).")
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        self.model = model
        self.name = f"deepseek:{model}"

    def __call__(self, text):
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(model=self.model, messages=build_messages(text),
                                                response_format={"type": "json_object"}, stream=False)
        dt = time.perf_counter() - t0
        raw = r.choices[0].message.content or ""
        u = getattr(r, "usage", None)
        return {"label": parse_label(raw), "raw": raw, "latency_s": dt,
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None)}


GBNF = 'root ::= "{\\"name\\": \\"" label "\\"}"\nlabel ::= ' + " | ".join(f'"{l}"' for l in LABELS)


class LlamaCppBackend:
    def __init__(self, gguf_path, n_ctx=2048, n_gpu_layers=-1, constrained=True):
        from llama_cpp import Llama, LlamaGrammar
        self.llm = Llama(model_path=str(gguf_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                         verbose=False)
        self.grammar = LlamaGrammar.from_string(GBNF) if constrained else None
        self.name = f"local:{os.path.basename(str(gguf_path))}{'' if constrained else ':free'}"

    def __call__(self, text):
        t0 = time.perf_counter()
        r = self.llm.create_chat_completion(messages=build_messages(text), temperature=0.0,
                                            max_tokens=64, grammar=self.grammar)
        dt = time.perf_counter() - t0
        raw = r["choices"][0]["message"]["content"] or ""
        return {"label": parse_label(raw), "raw": raw, "latency_s": dt,
                "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": r.get("usage", {}).get("completion_tokens")}
