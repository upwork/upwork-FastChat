import logging
import time

from ...config.constants import SCHEMA, KG_LLM_MODEL
from ...retriever import Context
from ...utils import llm_client

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_cypher_query_from_llm(context: Context) -> str:
    logging.info(f"Generating Cypher query using LLM: {KG_LLM_MODEL}")

    start_time = time.time()

    try:
        response = llm_client.chat.completions.create(
            model=KG_LLM_MODEL,
            messages=[
                {"role": "user", "content": context.parameters["text2cypher_prompt"]},
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
