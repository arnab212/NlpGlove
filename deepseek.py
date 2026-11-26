# Voice-to-Grasp - Experiment 1: DeepSeek zero-shot classification of the 96 test prompts
# (paper Sections 2.3 and 3.1).
#
# Sends each prompt with a fixed JSON prompt (scenario, available grasp strategies, output format)
# to the DeepSeek API and writes the predicted class to a CSV file (All_predictions/Experiment1).
# Note: the strategy descriptions in the prompt ("50/70/90% force", "+/-10%") only describe the
# classes to the model; the glove's actual set-points are given in the paper's Table 2.
# Needs : environment variable DEEPSEEK_API_KEY.
# Run   : python deepseek.py

import os
import json
import csv
from openai import OpenAI

# DeepSeek API configuration
api_key = os.environ.get("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com"
client = OpenAI(api_key=api_key, base_url=API_URL)

def command(command_text):
    system_prompt = "You are a grasp strategy selector for a soft robotic exoskeleton."

    prompt_data = {
        "scenario": "The user is wearing a soft robotic hand exoskeleton...",
        "exoskeleton_capabilities": {
            "finger_control": "Individual or grouped finger actuation for flexion and extension.",
            "assistance_levels": "Force levels: 50%, 70%, 90%.",
            "grasp_modifiers": ["tighten", "loosen"]
        },
        "available_grasp_strategies": {
            "0": {"name": "grasp", "description": "Flex all fingers at 70% force."},
            "1": {"name": "open_hand", "description": "Fully extend all fingers."},
            "2": {"name": "gentle_grasp", "description": "Flex all fingers at 50% force."},
            "3": {"name": "tight_grasp", "description": "Flex all fingers at 90% force."},
            "4": {"name": "pinch_grasp", "description": "Flex thumb and index finger at 70% force."},
            "5": {"name": "tighten", "description": "Increase grasp force by 10% (if already grasped)."},
            "6": {"name": "loosen", "description": "Decrease grasp force by 10% (if already grasped)."}
        },
        "input": command_text,
        "output_format": {
            "strategy_id": "id of the strategy to use, e.g., 2",
            "name": "name of the strategy, e.g., gentle_grasp",
            "confidence_score": "confidence score of the strategy, e.g., 0.91"
        },
        "api_version": "1.0"
    }

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the command context and strategy mapping:\n```json\n{json.dumps(prompt_data, indent=2)}\n```\nNow, analyze the example_input and return the best grasp strategy."}
        ],
        response_format={"type": "json_object"},
        stream=False
    )

    response_json = json.loads(response.choices[0].message.content)
    strategy_name = response_json.get("name", "no_strategy")
    print(f"Command: {command_text} -> Strategy: {strategy_name}")
    return strategy_name

# Batch process from file
if __name__ == "__main__":
    input_file = f"./commands"
    output_file = "prediction_deepseek.csv"

    with open(input_file, 'r') as f:
        commands = [line.strip() for line in f.readlines() if line.strip()]
        # commands =['grab the ball']

    with open(output_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Command", "Predicted_Strategy"])

        for cmd in commands:
            try:
                intent = command(cmd)
                writer.writerow([cmd, intent])
            except Exception as e:
                print(f"Error processing command '{cmd}': {e}")
                writer.writerow([cmd, "error"])
