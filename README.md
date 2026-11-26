# Voice-to-Grasp: a user-adaptive, language-based controller for an assistive soft robotic glove

This repository contains the code (and, once added, the data) for the paper
*"Voice-to-Grasp: A User-Adaptive Language-Based Controller for Assistive Robotic Gloves"*.

A spoken command is transcribed locally (Whisper), interpreted by a language model (a fine-tuned
DistilBERT on the device, or DeepSeek online) as one of seven grasp classes, and executed by a
two-channel pneumatic textile glove. The on-device model adapts to its user through continual
learning from the user's corrections.

| Grasp classes | `grasp`, `open_hand`, `gentle_grasp`, `tight_grasp`, `pinch_grasp`, `tighten`, `loosen` |
|---|---|

---

## 1. Repository structure

The folder layout is the one used in the study; the analysis scripts rely on it.

```
.
├── GuiUserStudyNew.py            Experiment 2: user-study GUI (speech -> both models -> glove -> yes/no feedback)
├── two_regulator_controller.py   PC side of the glove controller (serial, two set-points in %)
├── gloves_control/
│   └── gloves_control.ino        Arduino firmware (two PWM outputs to the pressure regulators)
├── deepseek.py                   Experiment 1: DeepSeek zero-shot predictions
├── confusionMatrix.py            Experiment 1: confusion matrices (original analysis)
├── modelTraining.ipynb           Fine-tuning of DistilBERT and RoBERTa (produces bert_finetuned_updated/)
├── Scripts/                      Experiment 1 baselines, dataset creation, original Exp. 2 analyses
│   ├── chatgpt.py  llama.py  predictions.py  accuracy.py
│   ├── datasetCreation.py  creatingSet.py  mergedata.py
│   ├── all_vizualizations.py  data_analysis.py  short_long_analysis.py
│   └── texttovoice.py
├── Scripts_ContinualLearning/    Experiment 3 (task specialisation) and the earlier continual learner
│   ├── continual_trainer.py  continual_batch_learning.py  continual_gui.py  mergeResultsUserstudy.py
│   ├── predictions.ipynb
│   └── analysis/Analysis.ipynb
├── Continual_Learning_Experiments/
│   └── UserStudy_code_analysis/  Experiment 4 (personalisation): final continual learner and analyses
│       ├── continual_trainer.py  continual_batch_learning.py
│       └── *.ipynb               combined, per-participant and cross-participant analyses
├── Revision_ISR_2026/            Scripts that produce every number and figure of the revised paper
│   ├── 01_data_audit.py ... 13_glove_actuation_from_video.py
│   └── common.py  cl_core.py  runner.py  llm_backends.py  safe_command.py
├── requirements.txt
└── requirements_study_environment.txt   full package list of the original study environment
```

Participants are identified only by the IDs used in the paper: **P1-P4** in Experiment 4 and
**user1-user20** in Experiment 2. Notebook outputs were cleared for this release.

## 2. Installation

Python 3.13 was used (Windows 11, CPU only).

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

`llama-cpp-python` is only needed for the offline LLM baselines (Llama-2, Mistral) and may need a
C++ compiler. Whisper needs `ffmpeg` only when transcribing audio files.

### API keys

No key is stored in the code. Set them as environment variables before running a script that calls an API:

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."     # PowerShell
$env:OPENAI_API_KEY   = "sk-..."     # only for Scripts/chatgpt.py
```
```bash
export DEEPSEEK_API_KEY="sk-..."     # bash
```

### Model weights

The weights are too large for GitHub (about 268 MB per model) and are provided separately:
**[link to be added]**. Place them as follows:

| Folder | Model | Used in |
|---|---|---|
| `bert_finetuned_updated/` | fine-tuned DistilBERT (base model) | Experiments 1-4, 2b, ablations |
| `Continual_Learning_Experiments/Tester-21/adaptive_model/` | kitchen-specialised model | Experiment 3 |
| `Continual_Learning_Experiments/P1 ... P4/adaptive_model/` | personalised models | Experiment 4 |

The GGUF files of Llama-2-7B-chat and Mistral-7B-Instruct (Q4_K_M) can be downloaded from their
public model pages; set the path in `Scripts/llama.py` or pass `--gguf` to `Revision_ISR_2026/06_llm_baselines.py`.

## 3. Data

| Data | Location | Content |
|---|---|---|
| Fine-tuning set | `dataset.json`, `updated_dataset.json` | 1,894 unique prompt-label pairs, 250-295 per class |
| Experiment 1 | `test_results.csv`, `All_predictions/` | 96 test prompts with the assessors' labels; predictions of the six models |
| Experiment 2 | `user_study/bert_results_userN.csv`, `deepseek_results_userN.csv` | 20 participants x 10 spoken commands: confirmed transcript, prediction, yes/no feedback, correction |
| Experiment 3 | `Scripts_ContinualLearning/*.xlsx`, `Continual_Learning_Experiments/Tester-*/` | kitchen/repair adaptation and test sets (standard and non-native); logs of the runs |
| Experiment 4 | `Continual_Learning_Experiments/UserStudy_code_analysis/VoiceToGrasp_StudySheet_P1..P4.xlsx`, `experiment_results/`, `cross_user_results/` | 210 written prompts per participant (140 adaptation, 70 test); results |
| Revision outputs | `Revision_ISR_2026/outputs/` | all recomputed numbers, tables and figures |

Audio was not stored during the study, so no voice recordings are included. Pilot sessions and
participants not reported in the paper are not included.

## 4. Hardware (Experiment 2)

* Textile soft robotic glove with fabric actuators, two pneumatic channels:
  channel 1 = thumb and index finger, channel 2 = the other three fingers.
* Arduino Mega running `gloves_control/gloves_control.ino`; PWM on pins 9 and 10 drives two
  SMC ITV2050 electronic pressure regulators supplied by an air compressor (maximum working pressure 69 kPa).
* PC to Arduino: USB serial, 115200 baud, one line `"p1 p2"` with both set-points in percent.
* Each grasp class sets a fixed pair of set-points (paper, Table 2); `open_hand` sets both to zero,
  `tighten`/`loosen` change the pressurised channels in steps of 25 %.

Upload the `.ino` file with the Arduino IDE, then set the serial port (e.g. `COM6`) in
`GuiUserStudyNew.py` (and `PORT` in `two_regulator_controller.py`).

## 5. Running the experiments

| Experiment (paper section) | Command |
|---|---|
| Fine-tuning (2.3) | run `modelTraining.ipynb` |
| Exp. 1, model comparison (2.3, 3.1) | `python deepseek.py`, `python Scripts/chatgpt.py`, `python Scripts/llama.py`, `python Scripts/predictions.py` |
| Exp. 2, user study (2.4, 3.2) | `python GuiUserStudyNew.py` (glove, microphone and `DEEPSEEK_API_KEY` required) |
| Exp. 3, task specialisation (2.6, 3.3) | `python Scripts_ContinualLearning/continual_batch_learning.py` (earlier continual learner) |
| Exp. 4, personalisation (2.7, 3.4) | `python Continual_Learning_Experiments/UserStudy_code_analysis/continual_batch_learning.py` (set `FILE_PATH` and `USER_ID`, e.g. `P1`) |

Some of the original scripts contain absolute Windows paths (for example `D:\PythonEnvs\nlpglove\...`);
edit them to point to your copy of the repository.

## 6. Reproducing the paper's numbers (`Revision_ISR_2026/`)

Run every script from inside `Revision_ISR_2026/`. They read the data folders (read-only) and write
only to `Revision_ISR_2026/outputs/`. Times are for a laptop CPU.

| Script | Produces (paper) | Time |
|---|---|---|
| `python 01_data_audit.py` | dataset size and class balance, train-test overlap, Experiment 2 measures (Sections 2.3, 2.9) | < 1 min |
| `python 02_paper_numbers.py` | Experiment 1 accuracies with 95 % Wilson intervals and McNemar test (Table 3); Experiment 4 base accuracies | ~1 min |
| `python 03_cl_ablations.py --suite lr --seeds 0 1 2 3 4` | ablation of the update strategy (Table 6) | ~1 min per run |
| `python 03_cl_ablations.py --suite buffer --seeds 0 1 2 3 4` | ablation of the replay buffer (Table 7) | ~1 min per run |
| `python 03_cl_ablations.py --suite noise --seeds 0 1 2 3 4` | wrong corrections, plausibility gate (Section 3.5) | ~1 min per run |
| `python 07_summarise_ablations.py` | summary tables and figures of all ablations | seconds |
| `python 04_exp2_adaptation.py --seeds 0 1 2` | Experiment 2b, adaptation on the spoken commands (Table 5) | ~30 min |
| `python 05_latency_benchmark.py --record 10 --port COM6 --deepseek-n 20` | latency of each stage (Table 8); needs microphone, glove and API key | ~20 min |
| `python 06_llm_baselines.py --backend deepseek --model deepseek-reasoner --sets exp4` | DeepSeek on the Experiment 4 test sets (Table 4) | ~10 min |
| `python 09_check_search_overlap.py` | no overlap between the learning-rate search prompts and Experiment 3 | seconds |
| `python 10_results_numbers.py` | Experiments 2-4: UCIM, McNemar and Wilcoxon tests, per-class F1, cross-participant matrix (Sections 3.2-3.4, Table 4) | a few minutes |
| `python 11_results_figures.py` | Figures 3-6 | ~1 min |
| `python 12_exp3_model_map.py` | which run and test files give the Experiment 3 numbers | a few minutes |
| `python 13_glove_actuation_from_video.py` | approximate actuation time from video (Table 8); the video is not included | ~1 min |
| `python safe_command.py` | self-test of the output validation (Section 2.2) | seconds |

The ablation runs can be stopped and resumed: finished (configuration, participant, seed) runs are skipped.

## 7. Notes

* The descriptions of the grasp strategies in the LLM prompts (e.g. "50/70/90 % force", "+/-10 %")
  only describe the classes to the model; the glove's set-points are those in the paper's Table 2.
* Experiment 3 used an earlier version of the continual learner (`Scripts_ContinualLearning/continual_trainer.py`);
  Experiment 4 and all ablations use the final version
  (`Continual_Learning_Experiments/UserStudy_code_analysis/continual_trainer.py`, re-implemented in `Revision_ISR_2026/cl_core.py`).
* The DeepSeek model behind the API changes over time; re-running the API scripts may give slightly different predictions.

## 8. Citation and licence

Citation and licence will be added after the review (this copy is anonymised for double-blind review).
