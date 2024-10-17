import openai
import config
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Set your OpenAI API key
client = OpenAI(api_key=config.OPENAI_API_KEY)

def create_natural_language_prompt(neptune_response, user_question):
    logging.info("Creating natural language prompt from Neptune result")

    nl_prompt = f"""
    Task: Convert the following data response from a graph database into a natural language explanation to response to user question:{user_question}
    Instructions: Based on the data provided, generate a concise and clear response based on the information in plain and conversational language.
    - If the data is a list, present it as a list.
    Data:
    {neptune_response}
    """

    logging.info(f"Created prompt: {nl_prompt[:200]}...")  # Print first 200 characters
    return nl_prompt


def convert_neptune_response_to_natural_language(nl_prompt):
    logging.info("Converting Neptune response to natural language")

    try:
        response_db = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.6,  # Adjust as needed
            messages=[
                {"role": "system", "content": "You are a data to language converter."},
                {"role": "user", "content": nl_prompt}
            ]
        )

        natural_language_response = response_db.choices[0].message.content.strip()
        logging.info(f"Converted natural language response: {natural_language_response[:200]}...")
        return natural_language_response

    except Exception as e:
        logging.error(f"Error during natural language conversion: {e}")
        raise
