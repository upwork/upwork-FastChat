from ..data_store import DataStore
from ..retriever import Results
import requests
import logging

HEADERS = {
    "Content-Type": "application/json",
}
logger = logging.getLogger(__name__)


class FreelancerProfileSemanticSearch(DataStore):
    DATA_STORE_NAME = "Freelancer Profile"

    def search(self, context) -> Results:
        profile_results = self._get_profile_results(context)
        return Results(objects=profile_results)

    def _get_profile_results(self, context) -> list:
        query = context.objects["query"]
        payload = {
            "index_name": "freelancer_profile_umrlarge_non_nested_demo",
            "field_to_search": "chunks_embeddings",
            "search_type": "vector_search",
            "top_k": 10,
            "query": query,
        }
        response = self._make_request(payload)
        results = [
            {
                "content": response["source_document"]["chunks"],
                "distance": response["distance"],
            }
            for response in response["responses"]
        ]
        sorted_results = sorted(results, key=lambda x: x["distance"])
        return sorted_results

    def _make_request(self, payload: dict) -> dict:
        response = requests.post(
            "https://umrsearchservice.umami.staging.platform.usw2.upwork/search/",
            headers=HEADERS,
            json=payload,
            verify=False,
        )
        return response.json()