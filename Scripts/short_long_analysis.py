# Voice-to-Grasp - Experiment 2: original analysis of accuracy versus command length.
# The revised paper's Fig. 3c (quartile groups, all 200 commands) comes from
# Revision_ISR_2026/11_results_figures.py.

# import pandas as pd
# import glob
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score

# data_folder = r'D:\PythonEnvs\nlpglove\user_study'
# csv_files = glob.glob(os.path.join(data_folder, '*.csv'))

# def get_gold_label(row):
#     if pd.notnull(row['correction']) and row['correction'] != '':
#         return row['correction']
#     elif str(row['feedback']).lower() == 'yes':
#         return row['prediction']
#     else:
#         return None

# # Compute global median sentence length
# all_lengths = []
# for f in csv_files:
#     df = pd.read_csv(f)
#     lengths = df['prompt'].apply(lambda x: len(str(x).split()))
#     all_lengths.extend(lengths)
# global_median_length = np.median(all_lengths)
# print(f"Global median sentence length: {global_median_length}")

# performance_by_length = []
# for f in csv_files:
#     fname = os.path.basename(f)
#     if fname.startswith('bert_'):
#         model = 'distillBERT'
#     elif fname.startswith('deepseek_'):
#         model = 'DeepSeek'
#     else:
#         continue
#     df = pd.read_csv(f)
#     df['sentence_length'] = df['prompt'].apply(lambda x: len(str(x).split()))
#     gold = [get_gold_label(row) for idx, row in df.iterrows()]
#     pred = df['prediction'].tolist()
#     filtered = [(g, p, l) for g, p, l in zip(gold, pred, df['sentence_length']) if g is not None]
#     if not filtered:
#         continue
#     gold_filtered, pred_filtered, len_filtered = zip(*filtered)
#     short_gold = [g for g, l in zip(gold_filtered, len_filtered) if l <= global_median_length]
#     short_pred = [p for p, l in zip(pred_filtered, len_filtered) if l <= global_median_length]
#     long_gold = [g for g, l in zip(gold_filtered, len_filtered) if l > global_median_length]
#     long_pred = [p for p, l in zip(pred_filtered, len_filtered) if l > global_median_length]
#     short_acc = accuracy_score(short_gold, short_pred) if short_gold else 0
#     long_acc = accuracy_score(long_gold, long_pred) if long_gold else 0
#     performance_by_length.append({
#         'user': fname.split('_')[-1].replace('.csv', ''),
#         'model': model,
#         'short_sentence_accuracy': short_acc,
#         'long_sentence_accuracy': long_acc
#     })

# perf_df = pd.DataFrame(performance_by_length)
# print(perf_df)
# perf_melted = perf_df.melt(id_vars=['user', 'model'],
#                            value_vars=['short_sentence_accuracy', 'long_sentence_accuracy'],
#                            var_name='sentence_length', value_name='accuracy')
# perf_melted['sentence_length'] = perf_melted['sentence_length'].replace({
#     'short_sentence_accuracy': 'Short', 'long_sentence_accuracy': 'Long'})
# perf_melted = perf_melted.dropna(subset=['accuracy'])
# perf_melted = perf_melted[(perf_melted['accuracy'] >= 0) & (perf_melted['accuracy'] <= 1)]

# plt.figure(figsize=(12, 7))
# sns.set(style="whitegrid")

# # Create boxplot with a nice color palette
# ax = sns.boxplot(
#     data=perf_melted,
#     x='sentence_length',
#     y='accuracy',
#     hue='model',
#     palette='Set2',
#     width=0.6,
#     fliersize=5,
#     linewidth=1.2
# )

# # Overlay actual data points with some jitter for clarity
# sns.stripplot(
#     data=perf_melted,
#     x='sentence_length',
#     y='accuracy',
#     hue='model',
#     palette='Set2',
#     dodge=True,
#     alpha=0.6,
#     size=6,
#     edgecolor='red',
#     linewidth=0.7,
#     ax=ax
# )

# # Remove the duplicate legend from stripplot, keep only one
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[0:len(set(perf_melted['model']))], labels[0:len(set(perf_melted['model']))], title='Model')

# # Add title and axis labels with larger font sizes
# ax.set_title('Model Accuracy by Sentence Length with Data Points', fontsize=16, weight='bold')
# ax.set_xlabel('Sentence Length', fontsize=14)
# ax.set_ylabel('Accuracy', fontsize=14)

# # Show grid for better readability
# ax.grid(True, which='major', linestyle='--', alpha=0.7)

# # Adjust tick label size
# ax.tick_params(axis='both', which='major', labelsize=12)

# plt.tight_layout()
# plt.show()

# # Custom palette for models
# custom_palette = {
#     'distillBERT': "#809fe0",  # bluish
#     'DeepSeek': "#ffad65",     # orangish
#     # Other models can use default colors or add here if needed
# }
# plt.figure(figsize=(12, 7))
# sns.set(style="whitegrid")

# # Boxplot with custom palette
# ax = sns.boxplot(
#     data=perf_melted,
#     x='sentence_length',
#     y='accuracy',
#     hue='model',
#     palette=custom_palette,
#     width=0.6,
#     fliersize=5,
#     linewidth=1.2
# )

# # Swarmplot with same palette colors
# sns.stripplot(
#     data=perf_melted,
#     x='sentence_length',
#     y='accuracy',
#     hue='model',
#     palette=custom_palette,
#     dodge=True,
#     alpha=0.6,
#     size=6,
#     edgecolor='red',
#     linewidth=0.7,
#     ax=ax
# )

# # Remove duplicate legend entries due to two plots with hue
# handles, labels = ax.get_legend_handles_labels()
# n_models = len(custom_palette)
# ax.legend(handles[:n_models], labels[:n_models], title='Model')

# # Titles and labels
# ax.set_title('Model Accuracy by Sentence Length with Data Points', fontsize=16, weight='bold')
# ax.set_xlabel('Sentence Length', fontsize=14)
# ax.set_ylabel('Accuracy', fontsize=14)

# # Grid style and tick font sizes
# ax.grid(True, which='major', linestyle='--', alpha=0.7)
# ax.tick_params(axis='both', which='major', labelsize=12)

# plt.tight_layout()
# plt.show()





# plt.figure(figsize=(12, 7))
# sns.set(style="whitegrid")
# sns.violinplot(
#     data=perf_melted,
#     x='sentence_length',
#     y='accuracy',
#     hue='model',
#     palette=custom_palette,
#     inner='quartile',
#     linewidth=1.2,
#     cut=0
# )
# plt.title('Model Accuracy Distribution by Sentence Length', fontsize=16, weight='bold')
# plt.xlabel('Sentence Length', fontsize=14)
# plt.ylabel('Accuracy', fontsize=14)
# plt.ylim(0, 1)
# plt.grid(True, which='major', linestyle='--', alpha=0.7)
# plt.legend(title="Model")
# plt.tight_layout()
# plt.show()


import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

data_folder = r'D:\PythonEnvs\nlpglove\user_study'
csv_files = glob.glob(os.path.join(data_folder, '*.csv'))

def get_gold_label(row):
    if pd.notnull(row['correction']) and row['correction'] != '':
        return row['correction']
    elif str(row['feedback']).lower() == 'yes':
        return row['prediction']
    else:
        return None

all_data = []
for f in csv_files:
    fname = os.path.basename(f)
    if fname.startswith('bert_'):
        model = 'distillBERT'
    elif fname.startswith('deepseek_'):
        model = 'DeepSeek'
    else:
        continue
    df = pd.read_csv(f)
    df['sentence_length_char'] = df['prompt'].apply(lambda x: len(str(x)))
    gold = [get_gold_label(row) for _, row in df.iterrows()]
    pred = df['prediction'].tolist()
    filtered = [(g, p, l) for g, p, l in zip(gold, pred, df['sentence_length_char']) if g is not None]
    for g, p, l in filtered:
        correct = int(g == p)
        all_data.append({
            'model': model,
            'sentence_length_char': l,
            'correct': correct
        })

df_all = pd.DataFrame(all_data)
# Filter sentences to length ≤ 90 characters
df_all = df_all[df_all['sentence_length_char'] <= 65]

# Compute accuracy per unique sentence length and model
agg_acc = df_all.groupby(['model', 'sentence_length_char']).agg(
    accuracy=('correct', 'mean'),
    count=('correct', 'size')
).reset_index()

# Plot scatter of accuracy per length, fit polynomial curve and confidence band
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

palette = {
    'distillBERT': '#4a7ebb',  # blue shade
    'DeepSeek': '#ff9900',     # orange shade
}

for model in agg_acc['model'].unique():
    model_data = agg_acc[agg_acc['model'] == model]
    x = model_data['sentence_length_char'].values
    y = model_data['accuracy'].values
    color = palette.get(model, 'gray')
    
    # Scatter plot of aggregated accuracy points (size proportional to count for visibility)
    sizes = model_data['count'].values
    # plt.scatter(x, y, alpha=0.7, s=sizes * 10, label=f"{model} Data", color=color)
    
    # Polynomial fit (degree 2)
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = p(x_fit)
    
    residuals = y - p(x)
    mse = np.mean(residuals**2)
    
    dof = max(0, len(x) - (len(z)))
    tval = t.ppf(0.975, dof) if dof > 0 else 1.96
    se = np.sqrt(mse * (1/len(x) + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2)))
    
    lower = y_fit - tval * se
    upper = y_fit + tval * se
    
    plt.plot(x_fit, y_fit, linestyle='--', color=color, label=f"{model} Fit")
    plt.fill_between(x_fit, lower, upper, color=color, alpha=0.2)

plt.title('Accuracy vs Sentence Length (Characters) Aggregated by Length', fontsize=16, weight='bold')
plt.xlabel('Sentence Length (Characters)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(title='Model')
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, plt.xlim()[1] + 1, 5))
plt.yticks(np.arange(0, 1.5, 0.05))


plt.tight_layout()
plt.show()

