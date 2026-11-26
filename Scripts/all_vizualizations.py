# Voice-to-Grasp - Experiment 2: original figures of the user study (confusion matrices,
# per-participant results, precision/recall/F1) from user_study/*.csv.
# The figures in the revised paper are produced by Revision_ISR_2026/11_results_figures.py.
# Set data_folder below to your user_study folder.

import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from adjustText import adjust_text
import matplotlib.colors as mcolors

# Folder path containing all user files
data_folder = r'D:\PythonEnvs\nlpglove\user_study'
csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
# print(len(csv_files))

all_labels = ['grasp', 'loosen', 'tight_grasp', 'open_hand', 'pinch_grasp', 'gentle_grasp', 'tighten']
display_labels = [label.replace('_', ' ') for label in all_labels]

def get_gold_label(row):
    if pd.notnull(row['correction']) and row['correction'] != '':
        return row['correction']
    elif str(row['feedback']).lower() == 'yes':
        return row['prediction']
    else:
        return None

# Store per-user metrics for both models
user_metrics = []

for f in csv_files:
    fname = os.path.basename(f)
    if fname.startswith('bert_'):
        model = 'distillBERT'
    elif fname.startswith('deepseek_'):
        model = 'DeepSeek'
    else:
        continue
    df = pd.read_csv(f)
    gold = [get_gold_label(row) for idx, row in df.iterrows()]
    pred = df['prediction'].tolist()
    filtered = [(g, p) for g, p in zip(gold, pred) if g is not None]
    if not filtered:
        continue
    gold_filtered, pred_filtered = zip(*filtered)
    metrics = {
        'user': fname.split('_')[-1].replace('.csv', '').replace('user', 'P'),
        'model': model,
        'accuracy': accuracy_score(gold_filtered, pred_filtered),
        'precision': precision_score(gold_filtered, pred_filtered, average=None, labels=all_labels, zero_division=0),
        'recall': recall_score(gold_filtered, pred_filtered, average=None, labels=all_labels, zero_division=0),
        'f1': f1_score(gold_filtered, pred_filtered, average=None, labels=all_labels, zero_division=0),
    }
    user_metrics.append(metrics)

metrics_df = pd.DataFrame(user_metrics)

num_metrics = 3
num_models = 2
total_width = 0.8  # total width for all bars in a group
bar_width = total_width / (num_metrics * num_models)
all_metrics = ['precision', 'recall', 'f1']
metric_labels = ['Precision', 'Recall', 'F1']


plt.figure(figsize=(16, 7))
sns.set(style="whitegrid")

palette = {
    'distillBERT': ['#4a7ebb', '#7fa7e9', '#3a5f9e'],  # blue shades
    'DeepSeek': ['#ff9900', '#ffc966', '#cc7a00'],     # orange shades
}

x = np.arange(len(all_labels))

for i, model in enumerate(['distillBERT', 'DeepSeek']):
    model_df = metrics_df[metrics_df['model'] == model]
    mean_metrics = [np.mean(np.vstack(model_df[m]), axis=0) for m in all_metrics]
    
    for j, metric in enumerate(all_metrics):
        # Calculate position for each bar:
        # start from left edge of group at (x - total_width/2)
        pos = x - total_width/2 + (i * num_metrics + j) * bar_width + bar_width/2
        plt.bar(pos, mean_metrics[j], width=bar_width, label=f'{model} {metric_labels[j]}', color=palette[model][j])

plt.xticks(x, display_labels, rotation=45, ha='right')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Per-class Precision, Recall, and F1 for distillBERT and DeepSeek')
plt.ylim(0, 1)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Aggregate metrics by model
for model in ['distillBERT', 'DeepSeek']:
    model_df = metrics_df[metrics_df['model'] == model]    
    mean_p = np.mean(np.vstack(model_df['precision']), axis=0)
    mean_r = np.mean(np.vstack(model_df['recall']), axis=0)
    mean_f1 = np.mean(np.vstack(model_df['f1']), axis=0)
    plt.figure(figsize=(10,6))
    inds = np.arange(len(all_labels))
    width = 0.25
    plt.bar(inds, mean_p, width, label='Precision')
    plt.bar(inds+width, mean_r, width, label='Recall')
    plt.bar(inds+2*width, mean_f1, width, label='F1')
    plt.xticks(inds+width, display_labels, rotation=90)
    plt.title(f'Per-class Metrics (Precision, Recall, F1) - {model}')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Convert metrics to heatmap format (mean over users per class/model)
heatmap_data = []
for model in ['distillBERT', 'DeepSeek']:
    model_df = metrics_df[metrics_df['model'] == model]
    mean_prec = np.mean(np.vstack(model_df['precision']), axis=0)
    mean_recall = np.mean(np.vstack(model_df['recall']), axis=0)
    mean_f1 = np.mean(np.vstack(model_df['f1']), axis=0)
    for i, label in enumerate(display_labels):
        heatmap_data.append({'Model': model, 'Class': label, 
                             'Precision': mean_prec[i], 
                             'Recall': mean_recall[i], 
                             'F1': mean_f1[i]})

heatmap_df = pd.DataFrame(heatmap_data)
pivoted = heatmap_df.pivot(index='Class', columns='Model', values='F1')

plt.figure(figsize=(8,6))
sns.heatmap(pivoted, annot=True, cmap='Blues', fmt='.2f')
plt.title('Per-Class F1 Score by Model')
plt.xlabel('Model')
plt.ylabel('Class')
plt.tight_layout()
plt.show()


# Boxplot of accuracy distribution per model
plt.figure(figsize=(10,6))
metrics_df.boxplot(column='accuracy', by='model')
plt.title('Accuracy Distribution Per Model')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.suptitle('')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,7))
sns.boxplot(x='model', y='accuracy', data=metrics_df, 
            palette='Set3', 
            width=0.5, 
            showfliers=True,
            boxprops={'facecolor':'#e0e4fd', 'edgecolor':'#3c62b4', 'linewidth':2},
            medianprops={'color':'#ff7f0e', 'linewidth':2})
sns.swarmplot(x='model', y='accuracy', data=metrics_df, 
              color='#595959', alpha=0.8, size=8)
plt.title('Accuracy Distribution Per Model', fontsize=18)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Define custom colors for models
custom_palette = {
    'distillBERT': "#809fe0",  # bluish
    'DeepSeek': "#ffad65",     # orangish
    # Other models can use default colors or add here if needed
}



plt.figure(figsize=(10,6))
metrics_df.boxplot(column='accuracy', by='model')
plt.title('Accuracy Distribution Per Model')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.suptitle('')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,7))
sns.boxplot(x='model', y='accuracy', data=metrics_df,
            palette=custom_palette,
            width=0.5,
            showfliers=True,
            boxprops={'edgecolor': "#0051ff", 'linewidth': 2},  # edge color basic
            medianprops={'color': "#fd7600", 'linewidth': 2})
sns.swarmplot(x='model', y='accuracy', data=metrics_df,
              color="#111111", alpha=0.8, size=8)
plt.title('Accuracy Distribution Per Model', fontsize=18)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# # Scatter plot: user-level BERT vs DeepSeek accuracy
df_pivot = metrics_df.pivot(index='user', columns='model', values='accuracy')
if 'distillBERT' in df_pivot and 'DeepSeek' in df_pivot:
    plt.figure(figsize=(7,7))
    plt.scatter(df_pivot['distillBERT'], df_pivot['DeepSeek'], c='purple')
    for idx, row in df_pivot.iterrows():
        plt.text(row['distillBERT'], row['DeepSeek'], idx, fontsize=9)
    plt.xlabel('distillBERT Accuracy')
    plt.ylabel('DeepSeek Accuracy')
    plt.title('User-level Accuracy Comparison: distillBERT vs DeepSeek')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Set style for aesthetics
# sns.set(style='whitegrid', context='notebook', font_scale=1.2)

# plt.figure(figsize=(8, 8))
# # Scatter points with gradient color and shadow
# points = plt.scatter(df_pivot['distillBERT'], df_pivot['DeepSeek'],
#                      s=120, c=df_pivot['distillBERT'] - df_pivot['DeepSeek'],
#                      cmap='coolwarm', alpha=0.9, edgecolor='black', linewidth=1.5)

# # Add identity line with bold style
# lims = [0, 1]
# plt.plot(lims, lims, '--', color='#ff9900', lw=0.5, alpha=0.7, label='y = x (identity)')

# # Annotate user labels with offset for clarity
# for idx, row in df_pivot.iterrows():
#     plt.annotate(idx,
#                  (row['distillBERT'], row['DeepSeek']),
#                  textcoords="offset points",
#                  xytext=(7, -7),
#                  ha='left', fontsize=11, weight='bold',
#                  bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.5))

# # Add colorbar to show accuracy difference
# cb = plt.colorbar(points, shrink=0.82)
# cb.set_label('Accuracy Difference (BERT - DeepSeek)', fontsize=13)

# plt.xlabel('distillBERT Accuracy', fontsize=15, weight='bold')
# plt.ylabel('DeepSeek Accuracy', fontsize=15, weight='bold')
# plt.title('User-level Accuracy: distillBERT vs DeepSeek', fontsize=17, weight='bold')
# plt.xlim(0.5, 1.05)
# plt.ylim(0.5, 1.05)
# plt.xticks(np.arange(0.5, 1.05, 0.05), fontsize=12)
# plt.yticks(np.arange(0.5, 1.05, 0.05), fontsize=12)
# plt.legend(loc='upper left', fontsize=13, frameon=True)
# plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.3)
# plt.tight_layout()
# plt.show()



# Assume df_pivot: index = user, columns = ['distillBERT', 'DeepSeek'] as task completion success rates

sns.set(style='whitegrid', context='notebook', font_scale=1.2)
plt.figure(figsize=(8, 8))

points = plt.scatter(df_pivot['distillBERT'], df_pivot['DeepSeek'],
                     s=120,
                     c=(df_pivot['DeepSeek'] - df_pivot['distillBERT']),  # Success diff DeepSeek - BERT
                     cmap='coolwarm',
                     alpha=0.9,
                     edgecolor='black',
                     linewidth=1.5)

lims = [0, 1]
plt.plot(lims, lims, '--', color='#ff9900', lw=2.5, alpha=0.7, label='y = x (identity)')

for idx, row in df_pivot.iterrows():
    plt.annotate(
        idx,
        (row['distillBERT'], row['DeepSeek']),
        textcoords="offset points",
        xytext=(7, -7),
        ha='left',
        fontsize=11,
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.5)
    )

cb = plt.colorbar(points, shrink=0.82)
cb.set_label('Success Rate Difference (DeepSeek - distillBERT)', fontsize=13)



plt.xlabel('distillBERT Task Success Rate', fontsize=15, weight='bold')
plt.ylabel('DeepSeek Task Success Rate', fontsize=15, weight='bold')
plt.title('User-level Task Completion Success: distillBERT vs DeepSeek', fontsize=17, weight='bold')
plt.xlim(0.0, 1.05)
plt.ylim(0.0, 1.05)
plt.xticks(np.arange(0, 1.05, 0.05), fontsize=12)
plt.yticks(np.arange(0, 1.05, 0.05), fontsize=12)
plt.legend(loc='upper left', fontsize=13, frameon=True)
plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.3)
plt.tight_layout()
plt.show()

sns.set(style='whitegrid', context='notebook', font_scale=1.2)
plt.figure(figsize=(8, 8))
points = plt.scatter(df_pivot['distillBERT'], df_pivot['DeepSeek'],
                     s=120,
                     c=(df_pivot['DeepSeek'] - df_pivot['distillBERT']),
                     cmap='coolwarm',
                     alpha=0.9,
                     edgecolor='black',
                     linewidth=1.5)
lims = [0, 1]
plt.plot(lims, lims, '--', color='#ff9900', lw=2.5, alpha=0.7, label='y = x (identity)')

texts = []
for idx, row in df_pivot.iterrows():
    texts.append(
        plt.text(row['distillBERT'], row['DeepSeek'], idx,
                 fontsize=11,
                 weight='bold',
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.5))
    )

# Automatically adjust text to minimize overlap with arrows pointing to points
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

cb = plt.colorbar(points, shrink=0.82)
cb.set_label('Success Rate Difference (DeepSeek - distillBERT)', fontsize=13)
plt.xlabel('distillBERT Task Success Rate', fontsize=15, weight='bold')
plt.ylabel('DeepSeek Task Success Rate', fontsize=15, weight='bold')
plt.title('User-level Task Completion Success: distillBERT vs DeepSeek', fontsize=17, weight='bold')
plt.xlim(0.3, 1.05)
plt.ylim(0.3, 1.05)
plt.xticks(np.arange(0.3, 1.05, 0.05), fontsize=12)
plt.yticks(np.arange(0.3, 1.05, 0.05), fontsize=12)
plt.legend(loc='upper left', fontsize=13, frameon=True)
plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.3)
plt.tight_layout()
plt.show()