import http.client as http_client
import logging
import re
import time

import requests

from .config import MAX_LIMIT, NEPTUNE_ENDPOINT, NEPTUNE_PORT, TIMEOUT_MS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_cypher_in_neptune(cypher_query):
    url = f"https://{NEPTUNE_ENDPOINT}:{NEPTUNE_PORT}/openCypher"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'query': cypher_query,
        'timeout': 7200000  # 30 minutes in milliseconds
    }
    logging.info(f"Sending query to Neptune: {cypher_query}")

    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, data=data)
        response_time = time.time() - start_time
        logging.info(f"Received response from Neptune in {response_time:.2f} seconds")
        if response.status_code == 200:
            try:
                logging.info(f"Raw Neptune response text: {response.text[:1000]}...")  # Log first 1000 chars
                response_json = response.json()
                
                logging.info(f"Full Neptune JSON response: {response_json}")
                
                # Check if the response is a dictionary and contains valid keys
                if isinstance(response_json, dict):
                    if 'results' in response_json:
                        return response_json['results']
                    else:
                        logging.error("Expected 'results' key not found in the response.")
                        return []
                elif isinstance(response_json, list):
                    # If it's a list, return it directly
                    return response_json
                else:
                    logging.error("Response is not in the expected format (neither dict nor list).")
                    return []
            except ValueError as e:
                # Log the specific parsing error and part of the response that caused it
                logging.error(f"Error parsing JSON from Neptune: {e}")
                logging.error(f"Raw response snippet: {response.text[:500]}...")  # Log first 500 chars of raw response
                return []
        else:
            logging.error(f"Neptune query failed with status {response.status_code}: {response.text}")
            raise Exception(f"Neptune query failed: {response.text}")
    except requests.Timeout:
        logging.error(f"Query timed out after {TIMEOUT_MS / 1000:.2f} seconds")
        raise
    except requests.RequestException as e:
        logging.error(f"Error with Neptune request: {e}")
        raise

def process_cypher_query(cypher_query):
    logging.info(f"Processing Cypher query: {cypher_query}")
    all_results = []
    skip = 0
    
    # Check if the query contains a LIMIT clause
    if 'LIMIT' in cypher_query.upper():
        logging.info("Query contains a LIMIT clause, executing as is.")
        return execute_cypher_in_neptune(cypher_query)
    else:
        logging.info("Query does not contain a LIMIT clause, adding pagination.")
        
        while True:
            paginated_query = f"{cypher_query} SKIP {skip} LIMIT {MAX_LIMIT}"
            results = execute_cypher_in_neptune(paginated_query)

            if not results or len(results) == 0:
                break

            all_results.extend(results)
            skip += MAX_LIMIT
            
            logging.info(f"Retrieved {len(results)} results, total so far: {len(all_results)}")

            # If the number of results returned is less than MAX_LIMIT, we know this is the last page
            if len(results) < MAX_LIMIT:
                logging.info("Reached the last page of results.")
                break

        return all_results
