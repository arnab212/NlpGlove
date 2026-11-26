# Voice-to-Grasp - Experiment 1: confusion matrices of the six models on the 96 test prompts
# (original analysis; paper Fig. 2). Reads test_results.csv and the prediction files.
# The recomputed accuracies with confidence intervals are produced by Revision_ISR_2026/02_paper_numbers.py.
# Run : python confusionMatrix.py

# import csv
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Read true labels (first label in columns 2:6 assumed primary)
# true_labels = []
# with open('test_results.csv', newline='', encoding='utf-8') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         true_labels.append(row[2])  # first correct label

# def read_predictions(filename):
#     preds = []
#     with open(filename, newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             preds.append(row[1])  # first predicted label
#     return preds

# # Read predicted labels per model
# pred_roberta = read_predictions('prediction_roberta_updated.csv')
# pred_distillBERT = read_predictions('prediction_bert_updated.csv')
# pred_chatGPT = read_predictions('prediction_chatGPT.csv')
# pred_chatGPT5 = read_predictions('prediction_chatGPT5.csv')
# pred_deepseek = read_predictions('prediction_deepseek.csv')
# pred_mistral = read_predictions('prediction_mistral.csv')
# pred_llama2 = read_predictions('prediction_llama2.csv')

# def plot_confusion_matrix(true_labels, pred_labels, model_name):
#     labels = sorted(list(set(true_labels) | set(pred_labels)))
#     # Remove underscores from labels for display
#     display_labels = [label.replace('_', ' ') for label in labels]
#     cm = confusion_matrix(true_labels, pred_labels, labels=labels)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title(f'Confusion Matrix for {model_name}')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()


# # Plot confusion matrices for all models
# plot_confusion_matrix(true_labels, pred_roberta, 'roBERTa')
# plot_confusion_matrix(true_labels, pred_distillBERT, 'distillBERT')
# plot_confusion_matrix(true_labels, pred_chatGPT, 'chatGPT')
# plot_confusion_matrix(true_labels, pred_chatGPT5, 'chatGPT5')
# plot_confusion_matrix(true_labels, pred_deepseek, 'DeepSeek')
# plot_confusion_matrix(true_labels, pred_mistral, 'Mistral')
# plot_confusion_matrix(true_labels, pred_llama2, 'llama2')


import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Read true labels (first label in columns 2:6 assumed primary)
true_labels = []
with open('test_results.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        true_labels.append(row[2])  # first correct label


def read_predictions(filename):
    preds = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            preds.append(row[1])  # first predicted label
    return preds


# Read predicted labels per model
pred_roberta = read_predictions('prediction_roberta_updated.csv')
pred_distillBERT = read_predictions('prediction_bert_updated.csv')
pred_chatGPT = read_predictions('prediction_chatGPT.csv')
pred_chatGPT5 = read_predictions('prediction_chatGPT5.csv')
pred_deepseek = read_predictions('prediction_deepseek.csv')
pred_mistral = read_predictions('prediction_mistral.csv')
pred_llama2 = read_predictions('prediction_llama2.csv')


def plot_confusion_matrix(true_labels, pred_labels, model_name, cmap):
    labels = sorted(list(set(true_labels) | set(pred_labels)))
    # Remove underscores from labels for display
    display_labels = [label.replace('_', ' ') for label in labels]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=cmap)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Define colormaps
orangish_cmap = plt.cm.Oranges
bluish_cmap = plt.cm.Blues

# Plot confusion matrices for all models with specified colormaps
plot_confusion_matrix(true_labels, pred_roberta, 'roBERTa', bluish_cmap)
plot_confusion_matrix(true_labels, pred_distillBERT, 'distillBERT', bluish_cmap)
plot_confusion_matrix(true_labels, pred_chatGPT, 'chatGPT', orangish_cmap)
plot_confusion_matrix(true_labels, pred_chatGPT5, 'chatGPT5', orangish_cmap)
plot_confusion_matrix(true_labels, pred_deepseek, 'DeepSeek', orangish_cmap)
plot_confusion_matrix(true_labels, pred_mistral, 'Mistral', greenish_cmap := plt.cm.Greens)
plot_confusion_matrix(true_labels, pred_llama2, 'llama2', greenish_cmap := plt.cm.Greens)
