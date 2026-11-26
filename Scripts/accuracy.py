# Voice-to-Grasp - Experiment 1: original accuracy comparison of the six models (bar chart).
# Kept for completeness. The values reported in the revised paper (with 95% Wilson intervals,
# McNemar test and correct alignment of the Llama-2 file) come from Revision_ISR_2026/02_paper_numbers.py.

# import csv
# import matplotlib.pyplot as plt

# # Initialize lists and counters
# correct_lists = []
# list_compare_roberta = []
# list_compare_distillBERT = []
# list_compare_chatGPT = []
# list_compare_chatGPT5 = []
# list_compare_deepseek = []
# list_compare_mistral = []
# list_compare_llama = []

# c_r = c_b = c_c = c_5 = c_d = c_m = c_l = 0

# # Load correct answers
# with open('test_results.csv', newline='', encoding='utf-8') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         correct_lists.append(row[2:6])
# t = len(correct_lists)

# # Function to compare predictions and count correct
# def count_correct(pred_file, compare_list, counter):
#     with open(pred_file, newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             compare_list.append(row[1:2])
#     for i in range(len(correct_lists)):
#         if any(p in compare_list[i] for p in correct_lists[i]):
#             counter += 1
#     return counter

# # Count correct predictions for each model
# c_r = count_correct('prediction_roberta_updated.csv', list_compare_roberta, c_r)
# c_b = count_correct('prediction_bert_updated.csv', list_compare_distillBERT, c_b)
# c_c = count_correct('prediction_chatGPT.csv', list_compare_chatGPT, c_c)
# c_5 = count_correct('prediction_chatGPT5.csv', list_compare_chatGPT5, c_5)
# c_d = count_correct('prediction_deepseek.csv', list_compare_deepseek, c_d)
# c_m = count_correct('prediction_mistral.csv', list_compare_mistral, c_m)
# c_l = count_correct('prediction_llama2.csv', list_compare_llama, c_l)

# # Calculate accuracy
# accuracy_roberta = (c_r / t) * 100 if t > 0 else 0
# accuracy_distillBERT = (c_b / t) * 100 if t > 0 else 0
# accuracy_chatGPT = (c_c / t) * 100 if t > 0 else 0
# accuracy_chatGPT5 = (c_5 / t) * 100 if t > 0 else 0
# accuracy_deepseek = (c_d / t) * 100 if t > 0 else 0
# accuracy_mistral = (c_m / t) * 100 if t > 0 else 0
# accuracy_llama = (c_l / t) * 100 if t > 0 else 0

# print(f"Accuracy for roBERTa: {accuracy_roberta:.2f}%")
# print(f"Accuracy for distillBERT: {accuracy_distillBERT:.2f}%")
# print(f"Accuracy for chatGPT: {accuracy_chatGPT:.2f}%")
# print(f"Accuracy for chatGPT5: {accuracy_chatGPT5:.2f}%")
# print(f"Accuracy for DeepSeek: {accuracy_deepseek:.2f}%")
# print(f"Accuracy for Mistral: {accuracy_mistral:.2f}%")
# print(f"Accuracy for llama2: {accuracy_llama:.2f}%")

# # Prepare data for plotting
# model_names = ['roBERTa', 'distillBERT', 'chatGPT', 'DeepSeek', 'Mistral', 'llama2']
# accuracy_values = [accuracy_roberta, accuracy_distillBERT, accuracy_chatGPT5,
#                    accuracy_deepseek, accuracy_mistral, accuracy_llama]

# # Bar plot
# plt.figure(figsize=(10, 6))
# plt.bar(model_names, accuracy_values, color='skyblue')
# plt.xlabel('Models')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy Comparison of Different Models')
# plt.ylim([0, 100])
# plt.grid(axis='y')
# plt.show()

# # # Pie chart
# # plt.figure(figsize=(8, 8))
# # plt.pie(accuracy_values, labels=model_names, autopct='%1.1f%%', startangle=140)
# # plt.title('Accuracy Distribution of Different Models')
# # plt.show()

# # # Line plot
# # plt.figure(figsize=(10, 6))
# # plt.plot(model_names, accuracy_values, marker='o', linestyle='-', color='green')
# # plt.xlabel('Models')
# # plt.ylabel('Accuracy (%)')
# # plt.title('Accuracy Trend Across Models')
# # plt.ylim([0, 100])
# # plt.grid(True)
# # plt.show()

# # # Scatter plot
# # plt.figure(figsize=(10, 6))
# # plt.scatter(model_names, accuracy_values, color='red')
# # plt.xlabel('Models')
# # plt.ylabel('Accuracy (%)')
# # plt.title('Accuracy Scatter Plot')
# # plt.ylim([0, 100])
# # plt.grid(True)
# # plt.show()

# # # Horizontal bar plot
# # plt.figure(figsize=(10, 6))
# # plt.barh(model_names, accuracy_values, color='purple')
# # plt.xlabel('Accuracy (%)')
# # plt.ylabel('Models')
# # plt.title('Horizontal Bar Plot of Accuracy')
# # plt.xlim([0, 100])
# # plt.grid(axis='x')
# # plt.show()

# # # Box plot
# # plt.figure(figsize=(8, 6))
# # plt.boxplot(accuracy_values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
# # plt.title('Box Plot of Model Accuracies')
# # plt.ylabel('Accuracy (%)')
# # plt.show()


import csv
import matplotlib.pyplot as plt

# Initialize lists and counters
correct_lists = []
list_compare_roberta = []
list_compare_distillBERT = []
list_compare_chatGPT = []
list_compare_chatGPT5 = []
list_compare_deepseek = []
list_compare_mistral = []
list_compare_llama = []
c_r = c_b = c_c = c_5 = c_d = c_m = c_l = 0

# Load correct answers
with open('test_results.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        correct_lists.append(row[2:6])
t = len(correct_lists)

# Function to compare predictions and count correct
def count_correct(pred_file, compare_list, counter):
    with open(pred_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            compare_list.append(row[1:2])
    for i in range(len(correct_lists)):
        if any(p in compare_list[i] for p in correct_lists[i]):
            counter += 1
    return counter

# Count correct predictions for each model
c_r = count_correct('prediction_roberta_updated.csv', list_compare_roberta, c_r)
c_b = count_correct('prediction_bert_updated.csv', list_compare_distillBERT, c_b)
c_c = count_correct('prediction_chatGPT.csv', list_compare_chatGPT, c_c)
c_5 = count_correct('prediction_chatGPT5.csv', list_compare_chatGPT5, c_5)
c_d = count_correct('prediction_deepseek.csv', list_compare_deepseek, c_d)
c_m = count_correct('prediction_mistral.csv', list_compare_mistral, c_m)
c_l = count_correct('prediction_llama2.csv', list_compare_llama, c_l)

# Calculate accuracy
accuracy_roberta = (c_r / t) * 100 if t > 0 else 0
accuracy_distillBERT = (c_b / t) * 100 if t > 0 else 0
accuracy_chatGPT = (c_c / t) * 100 if t > 0 else 0
accuracy_chatGPT5 = (c_5 / t) * 100 if t > 0 else 0
accuracy_deepseek = (c_d / t) * 100 if t > 0 else 0
accuracy_mistral = (c_m / t) * 100 if t > 0 else 0
accuracy_llama = (c_l / t) * 100 if t > 0 else 0

print(f"Accuracy for roBERTa: {accuracy_roberta:.2f}%")
print(f"Accuracy for distillBERT: {accuracy_distillBERT:.2f}%")
print(f"Accuracy for chatGPT: {accuracy_chatGPT:.2f}%")
print(f"Accuracy for chatGPT5: {accuracy_chatGPT5:.2f}%")
print(f"Accuracy for DeepSeek: {accuracy_deepseek:.2f}%")
print(f"Accuracy for Mistral: {accuracy_mistral:.2f}%")
print(f"Accuracy for llama2: {accuracy_llama:.2f}%")

# Define colormaps
bluish_cmap = plt.cm.Blues
orangish_cmap = plt.cm.Oranges
greenish_cmap = plt.cm.Greens

# Model names and accuracies for plotting
model_names = ['roBERTa', 'distillBERT', 'chatGPT', 'DeepSeek', 'Mistral', 'llama2']
accuracy_values = [accuracy_roberta, accuracy_distillBERT, accuracy_chatGPT5,
                   accuracy_deepseek, accuracy_mistral, accuracy_llama]

# Assign colors based on your scheme
colors = [
    bluish_cmap(0.6),      # roBERTa
    bluish_cmap(0.6),      # distillBERT
    # orangish_cmap(0.6),    # chatGPT
    orangish_cmap(0.6),    # chatGPT5
    orangish_cmap(0.6),    # DeepSeek
    greenish_cmap(0.6),    # Mistral
    greenish_cmap(0.6)     # llama2
]

# Bar plot
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_values, color=colors)
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of Different Models')
plt.ylim([0, 100])
plt.grid(axis='y')
plt.show()
