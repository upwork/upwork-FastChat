import os
import json
from openai import OpenAI
from .config.constants import LLM_CLIENT

PROMPT_DIR = "config/prompts"


def load_llm_client():
    if LLM_CLIENT == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif LLM_CLIENT == "fireworks":
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv("FIREWORKS_API_KEY"),
        )
    else:
        raise ValueError(f"Invalid LLM client: {LLM_CLIENT}")


llm_client = load_llm_client()


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
