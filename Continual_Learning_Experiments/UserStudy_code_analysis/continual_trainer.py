# Voice-to-Grasp - continual learner, FINAL version used in Experiment 4 (paper Section 2.5).
# Update on error: when the prediction is wrong, the corrected sample and up to 25 samples drawn from
# a 100-sample replay buffer (random replacement when full; every command is added) are used for one
# update. Dual learning rate: classification layer 9e-4, DistilBERT backbone 1e-5 (AdamW);
# the 768x768 pre-classifier layer is frozen.

# continual_trainer.py

import os
import json
import pickle
import random
import shutil
import csv
from datetime import datetime
import tempfile

import torch
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ID2LABEL = {
    0: "grasp",
    1: "open_hand",
    2: "gentle_grasp",
    3: "tight_grasp",
    4: "pinch_grasp",
    5: "tighten",
    6: "loosen"
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class ContinualTrainer:

    def __init__(self,
                 user_id,
                 base_model_path="./bert_finetuned_updated",
                 experiments_root="./Continual_Learning_Experiments",
                 replay_size=25,
                 buffer_size=100):
 
        self.user_id = user_id
        self.base_model_path = base_model_path
        self.user_root = os.path.join(experiments_root, user_id)
        self.adaptive_path = os.path.join(self.user_root, "adaptive_model")
        self.buffer_path = os.path.join(self.user_root, "replay_buffer.pkl")
        self.log_path = os.path.join(self.user_root, "log.csv")
        self.session_info_path = os.path.join(self.user_root, "session_info.json")

        self.replay_size = replay_size
        self.buffer_size = buffer_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._initialize_user_environment()

        # Load base model (frozen)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path).to(self.device)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Load adaptive model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.adaptive_path).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.adaptive_path)

        # Dual LR — backbone name varies by architecture (distilbert / bert / roberta …)
        backbone = None
        for attr in ("distilbert", "bert", "roberta", "electra", "deberta"):
            if hasattr(self.model, attr):
                backbone = getattr(self.model, attr)
                break

        param_groups = [{"params": self.model.classifier.parameters(), "lr": 9e-4}]
        if backbone is not None:
            param_groups.append({"params": backbone.parameters(), "lr": 1e-5})

        self.optimizer = AdamW(param_groups)

        self.replay_buffer = self._load_buffer()
        self.session_id = self._increment_session()

    # --------------------------------------------------
    # USER ENVIRONMENT SETUP
    # --------------------------------------------------

    def _initialize_user_environment(self):

        if not os.path.exists(self.user_root):
            os.makedirs(self.user_root)

        # If adaptive model doesn't exist → copy base
        if not os.path.exists(self.adaptive_path):
            shutil.copytree(self.base_model_path, self.adaptive_path)

        # Initialize replay buffer if not exists
        if not os.path.exists(self.buffer_path):
            with open(self.buffer_path, "wb") as f:
                pickle.dump([], f)

        # Initialize session info
        if not os.path.exists(self.session_info_path):
            info = {
                "total_sessions": 0,
                "total_updates": 0,
                "created_at": str(datetime.now())
            }
            with open(self.session_info_path, "w") as f:
                json.dump(info, f, indent=4)

        # Initialize log file
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "user_id",
                    "session_id",
                    "update_number",
                    "input_text",
                    "predicted_label",
                    "confidence",
                    "correct_label",
                    "loss",
                    "buffer_size"
                ])

    # --------------------------------------------------
    # SESSION MANAGEMENT
    # --------------------------------------------------

    def _increment_session(self):
        with open(self.session_info_path, "r") as f:
            info = json.load(f)

        info["total_sessions"] += 1
        current_session = info["total_sessions"]

        with open(self.session_info_path, "w") as f:
            json.dump(info, f, indent=4)

        return current_session

    def _increment_update_counter(self):
        with open(self.session_info_path, "r") as f:
            info = json.load(f)

        info["total_updates"] += 1
        update_number = info["total_updates"]

        with open(self.session_info_path, "w") as f:
            json.dump(info, f, indent=4)

        return update_number

    # --------------------------------------------------
    # REPLAY BUFFER
    # --------------------------------------------------

    def _load_buffer(self):
        with open(self.buffer_path, "rb") as f:
            return pickle.load(f)

    def _save_buffer(self):
        with open(self.buffer_path, "wb") as f:
            pickle.dump(self.replay_buffer, f)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    def predict(self, text):

        self.model.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()

        return ID2LABEL[pred_id], confidence

    # --------------------------------------------------
    # UPDATE
    # --------------------------------------------------

    def add_to_buffer(self, text, label):
        """
        Add any sample (correct or incorrect) to the replay buffer.
        Uses reservoir-style eviction: once full, randomly replace an
        existing entry so all classes stay represented over time.
        """
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append((text, label))
        else:
            replace_idx = random.randint(0, self.buffer_size - 1)
            self.replay_buffer[replace_idx] = (text, label)

    def update_model(self, text, correct_label):

        self.model.train()

        # Add to replay buffer (already added via add_to_buffer before this call)
        # — do NOT add again here to avoid duplicates

        replay_samples = random.sample(
            self.replay_buffer,
            min(self.replay_size, len(self.replay_buffer))
        )

        total_loss = 0

        for sample_text, label in replay_samples:
            inputs = self.tokenizer(
                sample_text,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)

            labels = torch.tensor([LABEL2ID[label]]).to(self.device)

            outputs = self.model(**inputs)
            loss = cross_entropy(outputs.logits, labels)
            total_loss += loss

        total_loss /= len(replay_samples)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Save everything
        self._save_buffer()
        # self.model.save_pretrained(self.adaptive_path)
        # self.tokenizer.save_pretrained(self.adaptive_path)
        temp_dir = self.adaptive_path + "_temp"

        # Remove temp if exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # Save to temp directory
        self.model.save_pretrained(temp_dir)
        self.tokenizer.save_pretrained(temp_dir)

        # Remove old model directory
        if os.path.exists(self.adaptive_path):
            shutil.rmtree(self.adaptive_path)

        # Rename temp → adaptive
        os.rename(temp_dir, self.adaptive_path)
        update_number = self._increment_update_counter()

        return total_loss.item(), update_number

    # --------------------------------------------------
    # LOGGING
    # --------------------------------------------------

    def log_event(self,
                  input_text,
                  predicted_label,
                  confidence,
                  correct_label,
                  loss,
                  update_number):

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                str(datetime.now()),
                self.user_id,
                self.session_id,
                update_number,
                input_text,
                predicted_label,
                confidence,
                correct_label,
                loss,
                len(self.replay_buffer)
            ])