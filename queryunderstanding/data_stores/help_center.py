from ..data_store import DataStore
from ..retriever import Results
import requests
import logging

HEADERS = {
    "Content-Type": "application/json",
}
logger = logging.getLogger(__name__)


class HelpCenterSemanticSearch(DataStore):
    DATA_STORE_NAME = "Help Center"

    def search(self, context) -> Results:
        qa_results = self._get_qa_results(context)
        return Results(objects=qa_results)

    def _get_qa_results(self, context) -> list:
        query = context.objects["query"]
        payload = {
            "index_name": "zendesk_q&a_umrlarge_non_nested",
            "field_to_search": "chunks_embeddings",
            "search_type": "vector_search",
            "top_k": context.parameters["help_center_top_k"],
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
        results = [
            {
                "content": result["content"],
            }
            for result in sorted(results, key=lambda x: x["distance"], reverse=True)
        ]
        return results

    def _make_request(self, payload: dict) -> dict:
        response = requests.post(
            "https://umrsearchservice.umami.staging.platform.usw2.upwork/search/",
            headers=HEADERS,
            json=payload,
            verify=False,
        )
        return response.json()
