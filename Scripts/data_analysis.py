# Voice-to-Grasp - Experiment 2: combined confusion matrices of DistilBERT and DeepSeek from
# user_study/*.csv (original analysis). Set data_folder below.

import pandas as pd
import glob
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Folder path containing all user files
data_folder = r'D:\PythonEnvs\nlpglove\user_study'
csv_files = glob.glob(os.path.join(data_folder, '*.csv'))

# All possible class labels
all_labels = ['grasp', 'loosen', 'tight_grasp', 'open_hand', 'pinch_grasp', 'gentle_grasp', 'tighten']
display_labels = [label.replace('_', ' ') for label in all_labels]
# Helper for gold label extraction
def get_gold_label(row):
    if pd.notnull(row['correction']) and row['correction'] != '':
        return row['correction']
    elif str(row['feedback']).lower() == 'yes':
        return row['prediction']
    else:
        return None

# Collect gold and predictions for BERT and DeepSeek across all users
gold_all = {'bert': [], 'deepseek': []}
pred_all = {'bert': [], 'deepseek': []}

for f in csv_files:
    fname = os.path.basename(f)
    if fname.startswith('bert_'):
        model = 'bert'
    elif fname.startswith('deepseek_'):
        model = 'deepseek'
    else:
        continue
    df = pd.read_csv(f)
    gold = [get_gold_label(row) for idx, row in df.iterrows()]
    pred = df['prediction'].tolist()
    # Only keep instances with a gold label
    filtered = [(g, p) for g, p in zip(gold, pred) if g is not None]
    gold_filtered, pred_filtered = zip(*filtered) if filtered else ([], [])
    gold_all[model].extend(gold_filtered)
    pred_all[model].extend(pred_filtered)

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot combined confusion matrix for BERT
cm_bert = confusion_matrix(gold_all['bert'], pred_all['bert'], labels=all_labels)
disp_bert = ConfusionMatrixDisplay(confusion_matrix=cm_bert, display_labels=display_labels)
disp_bert.plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Combined Confusion Matrix - distillBERT (All Users)')
axes[0].tick_params(axis='x', rotation=90)

# Plot combined confusion matrix for DeepSeek
cm_deepseek = confusion_matrix(gold_all['deepseek'], pred_all['deepseek'], labels=all_labels)
disp_deepseek = ConfusionMatrixDisplay(confusion_matrix=cm_deepseek, display_labels=display_labels)
disp_deepseek.plot(ax=axes[1], cmap='Oranges')
axes[1].set_title('Combined Confusion Matrix - DeepSeek (All Users)')
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()
