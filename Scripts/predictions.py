# Voice-to-Grasp - Experiment 1: predictions of the fine-tuned DistilBERT (./bert_finetuned_updated)
# on the 96 test prompts (paper Section 3.1). Writes the predicted class per prompt to CSV.
# Run : python Scripts/predictions.py   (from the repository root, so that the model path resolves)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import csv

id2label = {
    0: "grasp",
    1: "open_hand",
    2: "gentle_grasp",
    3: "tight_grasp",
    4: "pinch_grasp",
    5: "tighten",
    6: "loosen"
}

model_path = "./bert_finetuned_updated"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return id2label[predicted_class]


if __name__ == "__main__":    
    # print("\nEnter a command (e.g., 'grab the berry', 'release the berry'), or type 'exit' to quit.")
    # while True:
    #     user_input = input("\nCommand -> ")
    #     if user_input.lower() in ['exit', 'quit', 'q']:
    #         print("\nExiting and resetting hand...")            
    #         break
    #     intent = predict_intent(user_input)
    #     print(f"Detected intent: {intent}")
    file_path=f'./commands'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines=[line.strip() for line in lines]

    with open("prediction_bert_updated.csv", "w", newline ='') as cfile:
        writer = csv.writer(cfile)
        if lines:
         for line in lines:
            intent = predict_intent(line)
            print(f"command : {line} ------ Detected intent: {intent}")   
            writer.writerow([line,intent])
        else:
         print("No commands to execute")

    
        