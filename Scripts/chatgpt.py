# Voice-to-Grasp - Experiment 1: ChatGPT (GPT-5) zero-shot classification of the 96 test prompts
# (paper Sections 2.3 and 3.1). Same JSON prompt as deepseek.py; writes the predictions to CSV.
# Needs : environment variable OPENAI_API_KEY.
# Run   : python Scripts/chatgpt.py

import os




import json
from openai import OpenAI
import csv
import re

client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY")
) 

def command(command_text):
    prompt_data = {
        "scenario": "The user is wearing a soft robotic hand exoskeleton...",
        "exoskeleton_capabilities": {
            "finger_control": "Individual or grouped finger actuation for flexion and extension.",
            "assistance_levels": "Force levels: 50%, 70%, 90%.",
            "grasp_modifiers": ["tighten", "loosen"]
        },
        "available_grasp_strategies": {
            "0": { "name": "grasp", "description": "Flex all fingers at 70% force." },
            "1": { "name": "open_hand", "description": "Fully extend all fingers." },
            "2": { "name": "gentle_grasp", "description": "Flex all fingers at 50% force." },
            "3": { "name": "tight_grasp", "description": "Flex all fingers at 90% force." },
            "4": { "name": "pinch_grasp", "description": "Flex thumb and index finger at 70% force." },
            "5": { "name": "tighten", "description": "Increase grasp force by 10% (if already grasped)." },
            "6": { "name": "loosen", "description": "Decrease grasp force by 10% (if already grasped)." }
        },
        "input": command_text,
        "output_format": {
                "strategy_id": 2,
                "name": "gentle_grasp",
                "confidence_score": 0.91
            },
        "api_version": "1.0"
    }

    # response = client.chat.completions.create(
    #     model="gpt-5",
    #     messages=[
    #         {"role": "system", "content": "You are a grasp strategy selector for a soft robotic exoskeleton."},
    #         {"role": "user", "content": f"Here is the command context and strategy mapping:\n```json\n{json.dumps(prompt_data, indent=2)}\n```\nNow, analyze the example_input and return the best grasp strategy."}
    #     ],
    #     temperature=0.2
    # )

    user_payload = {
        "context": prompt_data,
        # "input": command_text,
        "instructions": "Return ONLY a JSON object with fields: strategy_id (int), name (string), confidence_score (float between 0 and 1)."
    }

    response = client.chat.completions.create(
        model="gpt-5",
        response_format={"type": "json_object"},  # requests strict JSON
        messages=[
            {"role": "system", "content": "You are a grasp strategy selector for a soft robotic exoskeleton."},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        # temperature=0.2,
        # max_output_tokens=100
    )


    print(response.choices[0].message.content)
    match = re.search(r'"name"\s*:\s*"([^"]+)"',response.choices[0].message.content)
    if match:
        strategy_name = match.group(1)
        print("Selected strategy:", strategy_name)
        return strategy_name
    else:
        print("No strategy name found.")
        return "no_strategy"




# Example usage
if __name__ == "__main__":        
    file_path=f'./commands'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines=[line.strip() for line in lines]
    # lines =['hold the bottle']

    with open("prediction_chatGPT5.csv", "w", newline ='') as cfile:
        writer = csv.writer(cfile)
        if lines:
         for line in lines:
            print(f"Processing command: {line}")
            intent = command(line)
            print(f"command : {line} ------ Detected intent: {intent}")   
            writer.writerow([line,intent])
        else:
         print("No commands to execute")

