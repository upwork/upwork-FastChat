from ..data_store import DataStore
from ..retriever import Results
import requests
import logging

HEADERS = {
    "Content-Type": "application/json",
}
logger = logging.getLogger(__name__)


class RemoteOpenSearch(DataStore):
    def connect(self):
        pass

    def search(self, context) -> Results:
        reviews = self._get_reviews(context.objects["query"], context.objects["freelancer_ids"])
        job_history = self._get_job_history(context.objects["query"], context.objects["freelancer_ids"])
        objects = reviews + job_history
        return Results(objects=objects)

    def _get_reviews(self, query: str, freelancer_ids: list[str]) -> Results:
        payload = {
            "index_name": "freelancer_job_review_umrlarge_non_nested",
            "field_to_search": "PROVIDER_COMMENT_EMBEDDINGS",
            "search_type": "filtered_vector_search",
            "filter_field_name": "PERSON_ID",
            "top_k": 10,
            "filter_field_values": freelancer_ids,
            "query": query,
        }
        response = self._make_request(payload)
        logger.info(f"Reviews: {response}")
        results = [
            {
                "content": response["source_document"]["PROVIDER_COMMENT"],
                "distance": response["distance"],
            }
            for response in response["responses"]
        ]
        sorted_results = sorted(results, key=lambda x: x["distance"])
        return sorted_results

    def _get_job_history(self, query: str, freelancer_ids: list[str]) -> Results:
        payload = {
            "index_name": "freelancer_job_history_umrlarge_non_nested",
            "field_to_search": "CONCAT_POST_TITLE_DESC_EMBEDDINGS",
            "search_type": "filtered_vector_search",
            "filter_field_name": "PERSON_ID",
            "top_k": 10,
            "filter_field_values": freelancer_ids,
            "query": query,
        }
        response = self._make_request(payload)
        logger.info(f"Job History: {response}")
        results = [
            {
                "content": response["source_document"]["CONCAT_POST_TITLE_DESC"],
                "distance": response["distance"],
            }
            for response in response["responses"]
        ]
        sorted_results = sorted(results, key=lambda x: x["distance"])
        return sorted_results

    def _make_request(self, payload: dict) -> Results:
        response = requests.post(
            "https://umrsearchservice.umami.staging.platform.usw2.upwork/search/",
            headers=HEADERS,
            json=payload,
            verify=False,
        )
        return response.json()
