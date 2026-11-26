# Voice-to-Grasp - Experiment 2 user-study GUI (paper Section 2.4).
#
# What it does
#   1. Records a spoken command (fixed 7-s window, 16 kHz, laptop microphone) and transcribes it
#      locally with OpenAI Whisper ("base" model).
#   2. Shows the transcript; the participant confirms it or speaks again.
#   3. Classifies the confirmed text with the fine-tuned DistilBERT (./bert_finetuned_updated)
#      and with DeepSeek (API, zero-shot, JSON output) into one of 7 grasp classes.
#   4. Executes each prediction on the glove, one after the other (buttons "Model 1"/"Model 2"),
#      through two_regulator_controller.py (two pneumatic channels, set-points in percent).
#   5. The participant answers yes/no and, if no, names the intended grasp.
# Outputs : user_study/bert_results_userN.csv and deepseek_results_userN.csv
#           (columns: prompt, prediction, feedback, correction). Edit the file names in
#           reset_iteration() for each participant.
# Needs   : environment variable DEEPSEEK_API_KEY; the glove's Arduino on a serial port
#           (set port= below); Windows (uses winsound); packages in requirements.txt.
# Run     : python GuiUserStudyNew.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pyttsx3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from PIL import Image, ImageTk, ImageSequence
import winsound
import json
from openai import OpenAI
import threading
import time
from two_regulator_controller import TwoRegulatorController

# ----- Whisper and audio dependencies -----
import whisper
import sounddevice as sd
import numpy as np
import queue

# --- Controller initialization (your settings) ---
ctrl = TwoRegulatorController(port='COM6', baud=115200)

# --- Load text classification model & tokenizer ----
MODEL_PATH = "./bert_finetuned_updated"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# --- DeepSeek/OpenAI setup (unchanged) ----
api_key = os.environ.get("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com"
client = OpenAI(api_key=api_key, base_url=API_URL)

def deepseek_predict(command_text):
    system_prompt = "You are a grasp strategy selector for a soft robotic exoskeleton."
    prompt_data = {
        "scenario": "The user is wearing a soft robotic hand exoskeleton...",
        "exoskeleton_capabilities": {
            "finger_control": "Individual or grouped finger actuation for flexion and extension.",
            "assistance_levels": "Force levels: 50%, 70%, 90%.",
            "grasp_modifiers": ["tighten", "loosen"]
        },
        "available_grasp_strategies": {
            "0": {"name": "grasp", "description": "Flex all fingers at 70% force."},
            "1": {"name": "open_hand", "description": "Fully extend all fingers."},
            "2": {"name": "gentle_grasp", "description": "Flex all fingers at 50% force."},
            "3": {"name": "tight_grasp", "description": "Flex all fingers at 90% force."},
            "4": {"name": "pinch_grasp", "description": "Flex thumb and index finger at 70% force."},
            "5": {"name": "tighten", "description": "Increase grasp force by 10% (if already grasped)."},
            "6": {"name": "loosen", "description": "Decrease grasp force by 10% (if already grasped)."}
        },
        "input": command_text,
        "output_format": {
            "strategy_id": "id of the strategy to use, e.g., 2",
            "name": "name of the strategy, e.g., gentle_grasp",
            "confidence_score": "confidence score of the strategy, e.g., 0.91"
        },
        "api_version": "1.0"
    }
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": (
        "Here is the command context and strategy mapping:\n"
        f"{json.dumps(prompt_data, indent=2)}\n"
        "Now, analyze the example_input and return the best grasp strategy. "
        "Respond only with a JSON object according to the output_format above."
    )},
]



    response = client.chat.completions.create(
        model="deepseek-reasoner",
        # messages=[
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": f"Here is the command context and strategy mapping:\n``````\nNow, analyze the example_input and return the best grasp strategy."}
        # ],
        messages=messages,
        response_format={"type": "json_object"},
        stream=False
    )
    response_json = json.loads(response.choices[0].message.content)
    strategy_name = response_json.get("name", "no_strategy")
    print(f"DeepSeek: {command_text} -> Strategy: {strategy_name}")
    return strategy_name

# --- BERT label mapping ----
ID2LABEL = {
    0: "grasp", 1: "open_hand", 2: "gentle_grasp",
    3: "tight_grasp", 4: "pinch_grasp", 5: "tighten", 6: "loosen"
}
def bert_predict(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
    print(f"Bert: {text} -> {ID2LABEL[idx]}")
    return ID2LABEL[idx]


class UserStudyGUI:
    def __init__(self, root):
        self.root = root
        root.title("User Study – Grasp Strategy Prediction")
        root.geometry("800x500")
        self.bert_df = pd.DataFrame(columns=["prompt", "prediction", "feedback", "correction"])
        self.deepseek_df = pd.DataFrame(columns=["prompt", "prediction", "feedback", "correction"])
        self.current_mode = None
        self.last_prediction = None
        self.text = ""
        self.deepseek_prediction = ""
        self.bert_prediction = ""

        # --- Panels and UI setup (unchanged) ---
        self.left_panel = tk.Frame(root, width=180, height=450)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_panel.pack_propagate(False)
        pil_gif = Image.open(r"D:\PythonEnvs\nlpglove\sound-8825.gif")
        self.gif_frames = [ImageTk.PhotoImage(
            frame.convert("RGBA").resize((200, 200), resample=Image.LANCZOS))
            for frame in ImageSequence.Iterator(pil_gif)]
        self.anim_label = tk.Label(self.left_panel)
        self.anim_label.place(relx=0.5, rely=0.05, anchor="n")
        self.animating = False
        self.current_frame = 0
        # ------ Listening timer label -----
        self.listen_running = False
        self.listen_start = 0
        self.listen_timer_label = tk.Label(self.left_panel, text="Listening time: 0.00 s", font=("Arial", 12))
        self.listen_timer_label.pack(pady=5)

        self.right = tk.Frame(root)
        self.right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.text_display = tk.Label(self.right, text="Detected Text:", font=("Arial", 14), wraplength=400, justify="left")
        self.text_display.pack(pady=10)
        self.start_button = tk.Button(self.right, text="Start Listening", command=self.start_listening)
        self.start_button.pack(pady=5)

        btn_frame = tk.Frame(self.right)
        btn_frame.pack(pady=5)
        self.yes_button = tk.Button(btn_frame, text="Yes", state=tk.DISABLED, command=self.on_yes)
        self.no_button = tk.Button(btn_frame, text="No", state=tk.DISABLED, command=self.on_no)
        self.yes_button.pack(side=tk.LEFT, padx=10)
        self.no_button.pack(side=tk.LEFT, padx=10)

        pred_frame = tk.Frame(self.right)
        pred_frame.pack(pady=10)
        self.loading = ttk.Progressbar(self.right, mode="indeterminate")
        self.bert_button = tk.Button(pred_frame, text="Perform Task by Model 1",
                                     state=tk.DISABLED, command=self.perform_task_bert)
        self.deepseek_button = tk.Button(pred_frame, text="Perform Task by Model 2",
                                         state=tk.DISABLED, command=self.perform_task_deepseek)
        self.bert_button.pack(side=tk.LEFT, padx=10)
        self.deepseek_button.pack(side=tk.LEFT, padx=10)
        self.bert_task = tk.Label(self.right, text="Model 1 task: ", font=("Arial", 14))
        self.bert_task.pack(pady=10)
        self.deepseek_task = tk.Label(self.right, text="Model 2 task: ", font=("Arial", 14))
        self.deepseek_task.pack(pady=10)

        ctrl_frame = tk.Frame(self.right)
        ctrl_frame.pack(pady=10)
        self.open_hand_button = tk.Button(ctrl_frame, text="Open Hand", command=self.open_hand)
        self.close_hand_button = tk.Button(ctrl_frame, text="Close Hand", command=self.close_hand)
        self.grasp_button = tk.Button(ctrl_frame, text="Grasp", command=self.grasp)
        self.pinch_button = tk.Button(ctrl_frame, text="Pinch", command=self.pinch_grasp)
        self.gentle_grasp_button = tk.Button(ctrl_frame, text="Gentle Grasp", command=self.gentle_grasp)
        self.tight_grasp_button = tk.Button(ctrl_frame, text="Tight Grasp", command=self.tight_grasp)
        self.tighten_button = tk.Button(ctrl_frame, text="Tighten", command=self.tighten)
        self.loosen_button = tk.Button(ctrl_frame, text="Loosen", command=self.loosen)
        self.open_hand_button.pack(side=tk.LEFT, padx=10)
        self.close_hand_button.pack(side=tk.LEFT, padx=10)
        self.grasp_button.pack(side=tk.LEFT, padx=10)
        self.pinch_button.pack(side=tk.LEFT, padx=10)
        self.gentle_grasp_button.pack(side=tk.LEFT, padx=10)
        self.tight_grasp_button.pack(side=tk.LEFT, padx=10)
        self.tighten_button.pack(side=tk.LEFT, padx=10)
        self.loosen_button.pack(side=tk.LEFT, padx=10)

        self.feedback_label = tk.Label(self.right, text="Were you satisfied?", font=("Arial", 12))
        self.feedback_label.pack(pady=5)
        feedback_frame = tk.Frame(self.right)
        feedback_frame.pack(pady=5)
        self.feedback_yes = tk.Button(feedback_frame, text="Yes", state=tk.DISABLED, command=self.store_feedback)
        self.feedback_no = tk.Button(feedback_frame, text="No", state=tk.DISABLED, command=self.ask_for_correction)
        self.feedback_yes.pack(side=tk.LEFT, padx=10)
        self.feedback_no.pack(side=tk.LEFT, padx=10)
        self.correction_var = tk.StringVar()
        self.correction_menu = ttk.Combobox(self.right, textvariable=self.correction_var,
            values=list(ID2LABEL.values()), state="readonly")
        self.corr_button = tk.Button(self.right, text="Confirm Correction", command=self.store_correction)
        end_reset = tk.Frame(self.right)
        end_reset.pack(pady=20)
        self.reset_button = tk.Button(end_reset, text="Reset Iteration", command=self.reset_iteration)
        self.end_button = tk.Button(end_reset, text="End Experiment", command=self.end_experiment)
        self.reset_button.pack(side=tk.LEFT, padx=20)
        self.end_button.pack(side=tk.LEFT, padx=20)

        self.pred_running = False
        self.pred_start = 0
        self.pred_timer_label = tk.Label(self.right, text="Prediction time: 0.00 s", font=("Arial", 12))
        self.pred_timer_label.pack(pady=5)

        # # ------ Listening timer label -----
        # self.listen_running = False
        # self.listen_start = 0
        # self.listen_timer_label = tk.Label(self.right, text="Listening time: 0.00 s", font=("Arial", 12))
        # self.listen_timer_label.pack(pady=5)

        # ---- Whisper model and audio -----
        self.whisper_model = whisper.load_model("base")  # or "tiny" for faster transcription
        self.audio_queue = queue.Queue()
        self.is_listening_whisper = False

        self.engine = pyttsx3.init()
        self.current_c1_percentage = 0
        self.current_c2_percentage = 0

    def _safe_send(self, p1, p2):
        try:
            ctrl.send_percents(p1, p2)
        except Exception as e:
            messagebox.showerror("Serial error", f"Failed to send to Arduino:\n{e}")

    # ----- Hand/Grasp Control -----
    def open_hand(self):
        self.current_c1_percentage = 0
        self.current_c2_percentage = 0
        self._safe_send(0, 0)

    def close_hand(self):
        # choose your “fully closed” level; 100% shown here
        self.current_c1_percentage = 100
        self.current_c2_percentage = 100
        self._safe_send(100, 100)

    def tighten(self):
        if self.current_c1_percentage != 0:
            self.current_c1_percentage = min(100, self.current_c1_percentage + 25)
        if self.current_c2_percentage != 0:
            self.current_c2_percentage = min(100, self.current_c2_percentage + 25)
        self._safe_send(self.current_c1_percentage, self.current_c2_percentage)

    def loosen(self):
        if self.current_c1_percentage != 0:
            self.current_c1_percentage = max(0, self.current_c1_percentage - 25)
        if self.current_c2_percentage != 0:
            self.current_c2_percentage = max(0, self.current_c2_percentage - 25)
        self._safe_send(self.current_c1_percentage, self.current_c2_percentage)

    def pinch_grasp(self):
        self.current_c1_percentage = 50
        self.current_c2_percentage = 0
        self._safe_send(50, 0)

    def grasp(self):
        self.current_c1_percentage = 45
        self.current_c2_percentage = 45
        self._safe_send(45, 45)

    def gentle_grasp(self):
        self.current_c1_percentage = 30
        self.current_c2_percentage = 30
        self._safe_send(30, 30)

    def tight_grasp(self):
        self.current_c1_percentage = 70
        self.current_c2_percentage = 70
        self._safe_send(70, 70)

    def perform_task(self, prediction):
        match prediction:
            case "grasp":
                self.grasp()
            case "open_hand":
                self.open_hand()
            case "gentle_grasp":
                self.gentle_grasp()
            case "tight_grasp":
                self.tight_grasp()
            case "pinch_grasp":
                self.pinch_grasp()
            case "tighten":
                self.tighten()
            case "loosen":
                self.loosen()

    # === WHISPER LISTENING ===
    def audio_callback(self, indata, frames, time_, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def listen_for_command_whisper(self):
        self.is_listening_whisper = True
        self.text_display.config(text="Listening (Whisper)...")
        sample_rate = 16000
        block_size = 1024
        audio_data = np.empty((0, 1), dtype=np.float32)
        start_time = time.time()
        max_duration = 7  # seconds
        self.listen_running = True
        self.listen_start = time.time()
        # Start audio input stream and collect audio
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=self.audio_callback, blocksize=block_size):
            while self.is_listening_whisper and (time.time() - start_time < max_duration):
                try:
                    data_chunk = self.audio_queue.get(timeout=0.1)
                    audio_data = np.concatenate((audio_data, data_chunk))
                except queue.Empty:
                    pass
        self.is_listening_whisper = False
        self.listen_running = False

        audio_data = audio_data.flatten()
        try:
            # Whisper expects float32 numpy array, 16000hz
            result = self.whisper_model.transcribe(audio_data, fp16=False, language="en")
            text = result.get("text", "").strip()
            if not text:
                text = "(couldn't understand)"
        except Exception as e:
            text = f"(Whisper error: {e})"
        self.root.after(0, lambda: self._on_listen_done(text))

    def start_listening(self):
        winsound.Beep(1000, 200)
        self.start_button.config(state=tk.DISABLED)
        self.text_display.config(text="Listening…")
        self.animating = True
        self.play_animation()
        self.listen_running = True
        self.listen_start = time.time()
        self._update_listen_timer()
        threading.Thread(target=self.listen_for_command_whisper, daemon=True).start()

    def _on_listen_done(self, text):
        self.listen_running = False
        self.listen_timer_label.config(text="Listening time: 0.00 s")
        self.animating = False
        self.anim_label.config(image="")
        self.text = text
        self.text_display.config(text=f"Detected: {text}")
        self.yes_button.config(state=tk.NORMAL)
        self.no_button.config(state=tk.NORMAL)

    def play_animation(self):
        if not self.animating:
            return
        frame = self.gif_frames[self.current_frame]
        self.anim_label.config(image=frame)
        self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
        self.root.after(100, self.play_animation)

    def _update_listen_timer(self):
        if not self.listen_running:
            return
        elapsed = time.time() - self.listen_start
        self.listen_timer_label.config(text=f"Listening time: {elapsed:.2f} s")
        self.root.after(100, self._update_listen_timer)

    def _update_pred_timer(self):
        if not self.pred_running:
            return
        elapsed = time.time() - self.pred_start
        self.pred_timer_label.config(text=f"Prediction time: {elapsed:.2f} s")
        self.root.after(100, self._update_pred_timer)

    def on_yes(self):
        self.yes_button.config(state=tk.DISABLED)
        self.no_button.config(state=tk.DISABLED)
        self.pred_start = time.time()
        self.pred_running = True
        self._update_pred_timer()
        self.loading.pack(pady=10)
        self.loading.start(10)
        threading.Thread(target=self._run_predictions, daemon=True).start()

    def on_no(self):
        self.text_display.config(text="Detected Text: ")
        self.start_button.config(state=tk.NORMAL)
        self.yes_button.config(state=tk.DISABLED)
        self.no_button.config(state=tk.DISABLED)

    def _run_predictions(self):
        bert_res = bert_predict(self.text)
        ds_res = deepseek_predict(self.text)
        self.root.after(0, lambda: self._on_predictions_done(bert_res, ds_res))

    def _on_predictions_done(self, bert_res, ds_res):
        self.loading.stop()
        self.loading.pack_forget()
        self.pred_running = False
        elapsed = time.time() - self.pred_start
        self.pred_timer_label.config(text=f"Prediction time: {elapsed:.2f} s")
        self.bert_prediction = bert_res
        self.deepseek_prediction = ds_res
        self.bert_button.config(state=tk.NORMAL)
        self.deepseek_button.config(state=tk.NORMAL)

    def perform_task_bert(self):
        self.current_mode = "bert"
        self.last_prediction = self.bert_prediction
        self.bert_task.config(text=f"Detected: {self.bert_prediction}")
        self.open_hand_button.config(state=tk.NORMAL)
        self.close_hand_button.config(state=tk.NORMAL)
        self.feedback_yes.config(state=tk.NORMAL)
        self.feedback_no.config(state=tk.NORMAL)
        self.perform_task(self.bert_prediction)

    def perform_task_deepseek(self):
        self.current_mode = "deepseek"
        self.last_prediction = self.deepseek_prediction
        self.deepseek_task.config(text=f"Detected: {self.deepseek_prediction}")
        self.open_hand_button.config(state=tk.NORMAL)
        self.close_hand_button.config(state=tk.NORMAL)
        self.feedback_yes.config(state=tk.NORMAL)
        self.feedback_no.config(state=tk.NORMAL)
        self.perform_task(self.deepseek_prediction)

    def store_feedback(self):
        df = self.bert_df if self.current_mode == "bert" else self.deepseek_df
        df.loc[len(df)] = [self.text, self.last_prediction, "yes", ""]
        messagebox.showinfo("Stored", "Feedback recorded.")
        self._post_feedback_cleanup()
    def ask_for_correction(self):
        self.correction_menu.pack(pady=5)
        self.corr_button.pack(pady=5)
    def store_correction(self):
        correction = self.correction_var.get()
        df = self.bert_df if self.current_mode == "bert" else self.deepseek_df
        df.loc[len(df)] = [self.text, self.last_prediction, "no", correction]
        messagebox.showinfo("Stored", "Correction recorded.")
        self.correction_menu.pack_forget()
        self.corr_button.pack_forget()
        self._post_feedback_cleanup()
    def _post_feedback_cleanup(self):
        for w in (self.feedback_yes, self.feedback_no):
            w.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
    def reset_iteration(self):
        for w in (self.yes_button, self.no_button,
                  self.bert_button, self.deepseek_button,
                  self.feedback_yes, self.feedback_no):
            w.config(state=tk.DISABLED)
        self.text_display.config(text="Detected Text:")
        self.bert_task.config(text="Model 1 task: ")
        self.deepseek_task.config(text="Model 2 task: ")
        self.correction_menu.pack_forget()
        self.corr_button.pack_forget()
        self.start_button.config(state=tk.NORMAL)
        self.pred_timer_label.config(text="Prediction time: 0.00 s")
        self.listen_timer_label.config(text="Listening time: 0.00 s")
    def end_experiment(self):
        self.bert_df.to_csv(r"D:\PythonEnvs\nlpglove\user_study\bert_results_user21_0.csv", index=False)
        self.deepseek_df.to_csv(r"D:\PythonEnvs\nlpglove\user_study\deepseek_results_user21_0.csv", index=False)
        messagebox.showinfo("Done", "Experiment data saved. Exiting.")
        self._safe_send(0, 0)
        ctrl.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    UserStudyGUI(root)
    root.mainloop()
