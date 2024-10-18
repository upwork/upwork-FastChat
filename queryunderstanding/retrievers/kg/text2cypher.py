import pandas as pd
import os
import logging
import time
from datetime import date
from openai import OpenAI

from ...config.constants import SCHEMA, KG_LLM_MODEL
from ...utils import load_prompt

today = date.today()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def create_cypher_query(context):
    logging.info(f"Generating Cypher query using LLM: {KG_LLM_MODEL}")

    start_time = time.time()

    try:
        prompt_template = load_prompt("text2cypher.txt")
        prompt = prompt_template.format(
            schema=SCHEMA,
            messages=context.messages,
            freelancers=context.objects["freelancers"],
            today=today,
        )
        response = client.chat.completions.create(
            model=KG_LLM_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a text-to-Cypher converter."},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        logging.error(f"Error in generating Cypher query: {e}")
        raise

    response_time = time.time() - start_time
    logging.info(f"Cypher query generated successfully in {response_time:.2f} seconds")

    # Log the full response for debugging
    logging.info(f"LLM Response: {response}")

    if response and response.choices:
        cypher_query = response.choices[0].message.content.strip()
        logging.info(f"Generated Cypher query: {cypher_query}")
        return cypher_query
    else:
        logging.error("No choices returned from LLM response")
        raise ValueError("LLM response did not return any choices.")
