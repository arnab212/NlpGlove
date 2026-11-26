# Voice-to-Grasp - small Tkinter GUI for interactive continual learning: type a command, see the
# prediction, confirm or correct it; corrections update the model (continual_trainer.py).

# continual_gui.py

import tkinter as tk
from tkinter import ttk, messagebox
from continual_trainer import ContinualTrainer, ID2LABEL


class ContinualGUI:

    def __init__(self, root):
        self.root = root
        root.title("User-Specific Continual Learning")
        root.geometry("600x500")

        self.trainer = None
        self.current_text = ""
        self.current_prediction = ""
        self.current_confidence = 0

        # ----------------------
        # USER LOGIN
        # ----------------------

        self.login_frame = tk.Frame(root)
        self.login_frame.pack(pady=20)

        tk.Label(self.login_frame, text="Enter User ID:").pack()

        self.user_entry = tk.Entry(self.login_frame)
        self.user_entry.pack(pady=5)

        tk.Button(self.login_frame,
                  text="Start Session",
                  command=self.start_session).pack(pady=5)

        # ----------------------
        # MAIN FRAME
        # ----------------------

        self.main_frame = tk.Frame(root)

        self.text_entry = tk.Entry(self.main_frame, width=60)
        self.text_entry.pack(pady=10)

        self.predict_button = tk.Button(
            self.main_frame,
            text="Predict",
            command=self.predict
        )
        self.predict_button.pack()

        self.pred_label = tk.Label(self.main_frame, text="Prediction: ")
        self.pred_label.pack(pady=10)

        self.feedback_label = tk.Label(self.main_frame, text="Was it correct?")
        self.feedback_label.pack()

        self.yes_button = tk.Button(
            self.main_frame,
            text="Yes",
            command=self.correct
        )
        self.no_button = tk.Button(
            self.main_frame,
            text="No",
            command=self.incorrect
        )

        self.yes_button.pack(side=tk.LEFT, padx=30)
        self.no_button.pack(side=tk.LEFT)

        self.correction_menu = ttk.Combobox(
            self.main_frame,
            values=list(ID2LABEL.values()),
            state="readonly"
        )

        self.update_button = tk.Button(
            self.main_frame,
            text="Update Model",
            command=self.update_model
        )

        self.status_label = tk.Label(self.main_frame, text="")

    # --------------------------------------------------

    def start_session(self):

        user_id = self.user_entry.get().strip()

        if not user_id:
            messagebox.showerror("Error", "User ID required.")
            return

        self.trainer = ContinualTrainer(user_id=user_id)

        self.login_frame.pack_forget()
        self.main_frame.pack(pady=10)

    # --------------------------------------------------

    def predict(self):

        text = self.text_entry.get().strip()
        if not text:
            return

        self.current_text = text

        pred, conf = self.trainer.predict(text)

        self.current_prediction = pred
        self.current_confidence = conf

        self.pred_label.config(
            text=f"Prediction: {pred} (confidence={conf:.3f})"
        )

        # Log prediction event (no update yet)
        self.trainer.log_event(
            input_text=text,
            predicted_label=pred,
            confidence=conf,
            correct_label="",
            loss="",
            update_number=""
        )

    # --------------------------------------------------

    def reset_screen(self):

        # Clear entry field
        self.text_entry.delete(0, tk.END)

        # Clear prediction label
        self.pred_label.config(text="Prediction: ")

        # Clear correction menu
        self.correction_menu.set("")
        self.correction_menu.pack_forget()

        # Hide update button
        self.update_button.pack_forget()

        # Clear status label
        self.status_label.config(text="")

        # Reset internal variables
        self.current_text = ""
        self.current_prediction = ""
        self.current_confidence = 0

    # --------------------------------------------------

    def correct(self):
        messagebox.showinfo("Stored", "Prediction accepted.")
        # Reset everything
        self.reset_screen()

    def incorrect(self):
        self.correction_menu.pack(pady=5)
        self.update_button.pack(pady=5)

    def update_model(self):

        correct_label = self.correction_menu.get()

        if not correct_label:
            return

        loss, update_number = self.trainer.update_model(
            self.current_text,
            correct_label
        )

        self.trainer.log_event(
            input_text=self.current_text,
            predicted_label=self.current_prediction,
            confidence=self.current_confidence,
            correct_label=correct_label,
            loss=loss,
            update_number=update_number
        )

        self.status_label.config(
            text=f"Model updated. Loss={loss:.4f} | Update #{update_number}"
        )
        self.status_label.pack(pady=10)

        self.correction_menu.pack_forget()
        self.update_button.pack_forget()

        self.reset_screen()


if __name__ == "__main__":
    root = tk.Tk()
    app = ContinualGUI(root)
    root.mainloop()