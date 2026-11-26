# Voice-to-Grasp - Experiment 1: offline LLM baseline (Llama-2-7B-chat, GGUF, llama-cpp-python)
# on the 96 test prompts (paper Sections 2.3 and 3.1). The Mistral-7B baseline uses the same
# script with model_path pointing to the Mistral GGUF file.
# Needs : the GGUF model file (download from its public model page; set model_path below).
# Output: prediction CSV; outputs that are not one of the 7 classes are counted as wrong.
# Run   : python Scripts/llama.py

import json
import csv
from llama_cpp import Llama

# Load local LLaMA 2 model
llm = Llama(
    model_path=r"D:\PythonEnvs\nlpglove\models\llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=20  # Optional if using GPU
)

def extract_json(text):
    """
    Try to extract a valid JSON object from the model's text output.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw output:\n{text}")

def command(command_text):
    system_prompt = (
        "You are a grasp strategy selector for a soft robotic exoskeleton. "
        "You will be given a command and a list of available grasp strategies."
        "Think step by step about which strategy is best suited for the command."
        "Consider the scenario, finger control, and force level needed."
        "Return your answer as a valid JSON object with the fields: "
        "`strategy_id`, `name of strategy`, and `confidence_score` (between 0 and 1)."
    )

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
            "strategy_id": "Id of the strategy",
            "name": "Name of the strategy",
            "confidence_score": "Confidence score"
        },
        "api_version": "1.0"
    }

    full_prompt = f"""### System:\n{system_prompt}\n\n### User:\nHere is the command context:\n{json.dumps(prompt_data, indent=2)}\n\nReturn the best grasp strategy as a JSON object:"""

    output = llm(full_prompt, max_tokens=512, temperature=0.2, stop=["</s>"])
    output_text = output["choices"][0]["text"].strip()
    print(f"Model Output: {output_text}")

    response_json = extract_json(output_text)
    strategy_name = response_json.get("name", "no_strategy")

    print(f"Command: {command_text} -> Strategy: {strategy_name}")
    return strategy_name

# Batch process from file
if __name__ == "__main__":
    input_file = "./commands"
    output_file = "prediction_llama2.csv"

    with open(input_file, 'r') as f:
        commands = [line.strip() for line in f.readlines() if line.strip()]

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
    
