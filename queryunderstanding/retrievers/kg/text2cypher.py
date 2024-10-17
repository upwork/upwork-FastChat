import pandas as pd
from openai import OpenAI
import os
import logging
import time
from datetime import date

from .config import SCHEMA, LLM_MODEL

today = date.today()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def create_cypher_prompt(context):
    """
    Creates a prompt to convert a natural language question into a Cypher query.
    :param user_question: The natural language question from the user.
    :return: A formatted prompt for the LLM.
    """
    schema = SCHEMA  # Use schema from the config

    logging.info("Creating Cypher prompt")
    prompt = f"""
    Task: Generate an openCypher statement to query a graph database based on the user question.
    Instructions: Based on the graph schema provided, generate a concise and efficient Cypher query.

    Guidelines:
    - Make sure you follow openCypher syntax.
    - Keep queries as simple as possible as the use of keywords is causing issues.
    - Use the provided relationship types and properties from the schema and make sure to follow the correct edge direction.
    - when using ids or labels make sure to treat them as strings with ''.
    - You cannot use aggregates inside aggregates.
    - Try to avoid aggregates if possible.
    - Avoid using collect and similar keywords.
    - Ensure that filtering is done using çWHERE clauses, and avoid using filters inside node patterns.
    - Return only the query, no comments, no explanations, no punctuation of any kind. No '''cypher, no '''
    - Structure the query with MATCH statements to traverse the graph.
    - Only return what is asked. If edges are requested, only return them. If nodes are requested, only return nodes.
    - All sub queries in an UNION must have the same column names.
    - For generic queries like outgoing/incoming edges keep it generic. 
            -outgoing MATCH (n {{id: 'Node'}})-[r]->(m) RETURN distinct type(r)
            -incoming MATCH (m)-[r]->(n {{id: 'Node'}}) RETURN distinct type(r)

    - Keep queries as simple as possible.
    - For creation time, job posts have ctime, freelancers have creation_date
    - ALWAYS use multimatch clauses with intermediate WITH statements instead of paths longer than 2 hops - for example use
            MATCH (c:Category)
            WHERE c.pref_label = 'Graphic Design'
            WITH c
            MATCH (sp:SpecializedProfile)-[:has_category]->(c)
            WITH sp, c
            MATCH (f:Freelancer)-[:has_profile]->(sp)

            instead of 

            MATCH (f:Freelancer)-[:has_profile]->(sp:SpecializedProfile)-[:has_category]->(c:Category)
            WHERE c.pref_label = 'Graphic Design'
            start from the smaller set and try to break every query with more than one steps
                 - i.e. MATCH (a:a)-[r1]->(b:b)-[r2]->(c:c) should be MATCH (a:a)-[r1]->(b:b) WITH a,b MATCH (b)-[r2]->(c:c)
                 - Make sure each match clause only has 2 steps (a)-[b]->(c)
    - Categories are organised into a taxonomy through relationship has_child, has_parent
    - Category taxonomy levels are defined via taxonomy_level property (CATEGORY, SUBCATEGORY, SERVICE)
    - The business jargon for these levels is CATEGORY=L1, SUBCATEGORY=L2, SERVICE=L3 
    - Skills are seeded to categories (L3) through AGs. (Skill)-[belongs_to_ag]->(AG), (AG)<-[related_ag]-(Category) 
    - If a summary of an entity is requested, give a short summary of key points and check 1 hop edges to gather more information.
    - When dealing with dates, use datetime() only for string literals, not variables. c.date>datetime("2020-01-01T00:00:00"). Do not use datetime in other elements like epochMillis.
    - When dealing with duration, i.e. within 5 days, convert date properties to epochMillis(f.date) instead of using  duration.between(datetime(f.date).
    - Avoid using duration() and other date related functions.
    - Avoid using head keyword.
    - Property access over temporal types is not supported.
    - When you want to compare entities to check if they are different (<>), use a property like uid and not the full node. e.g. a.uid <> b.uid instead of a<>b.
    - If you need to do date calculations, today is {today}.

    ### Schema
    {schema}

    ### User Question
    The user asks:
    {context.messages}

    ### Freelancer IDs
    {context.objects["freelancer_ids"]}
    """
    
    logging.info(f"Cypher prompt created: {prompt[:200]}...")  # Print first 200 characters
    return prompt

def generate_cypher_query_from_llm(prompt):
    """
    Generate a Cypher query using OpenAI's ChatCompletion API based on the prompt.
    :param prompt: The prompt to send to the LLM.
    :return: Generated Cypher query.
    """
    logging.info(f"Generating Cypher query from LLM: {LLM_MODEL}")
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.1,  # Adjust as needed
            messages=[
                {"role": "system", "content": "You are a text-to-Cypher converter."},
                {"role": "user", "content": prompt}
            ]
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