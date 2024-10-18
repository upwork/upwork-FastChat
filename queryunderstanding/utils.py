import os
import json

PROMPT_DIR = "config/prompts"


def load_prompt(prompt_file):
    """
    Loads the prompt from a .txt file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(current_dir, PROMPT_DIR, prompt_file)
    with open(prompt_file_path, "r") as file:
        return file.read()


def load_freelancers():
    """
    Loads the freelancers from a .json file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    freelancers_file_path = os.path.join(current_dir, "data/freelancers.json")
    with open(freelancers_file_path, "r") as file:
        return json.load(file)
