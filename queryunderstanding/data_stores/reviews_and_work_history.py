from ..data_store import DataStore
from ..retriever import Results
import requests
import logging

HEADERS = {
    "Content-Type": "application/json",
}
logger = logging.getLogger(__name__)


class ReviewsAndWorkHistorySemanticSearch(DataStore):
    DATA_STORE_NAME = "Reviews and Work History"

    def search(self, context) -> Results:
        reviews = self._get_reviews(context)
        job_history = self._get_job_history(context)
        objects = reviews + job_history
        return Results(objects=objects)

    def _get_reviews(self, context) -> Results:
        freelancers = {
            freelancer["person_id"]: freelancer["name"]
            for freelancer in context.objects["freelancers"]
        }
        query = context.objects["query"]
        payload = {
            "index_name": "freelancer_job_review_umrlarge_non_nested",
            "field_to_search": "PROVIDER_COMMENT_EMBEDDINGS",
            "search_type": "filtered_vector_search",
            "filter_field_name": "PERSON_ID",
            "top_k": context.parameters["reviews_top_k"],
            "filter_field_values": list(freelancers.keys()),
            "query": query,
        }
        response = self._make_request(payload)
        results = [
            {
                "person_id": response["source_document"]["PERSON_ID"],
                "content": response["source_document"]["PROVIDER_COMMENT"],
                "distance": response["distance"],
                "name": freelancers[str(response["source_document"]["PERSON_ID"])],
            }
            for response in response["responses"]
        ]
        results = [
            {"name": result["name"], "content": result["content"]}
            for result in sorted(results, key=lambda x: x["distance"], reverse=True)
        ]
        return results

    def _get_job_history(self, context) -> Results:
        freelancers = {
            freelancer["person_id"]: freelancer["name"]
            for freelancer in context.objects["freelancers"]
        }
        query = context.objects["query"]
        payload = {
            "index_name": "freelancer_job_history_umrlarge_non_nested",
            "field_to_search": "CONCAT_POST_TITLE_DESC_EMBEDDINGS",
            "search_type": "filtered_vector_search",
            "filter_field_name": "PERSON_ID",
            "top_k": context.parameters["job_history_top_k"],
            "filter_field_values": list(freelancers.keys()),
            "query": query,
        }
        response = self._make_request(payload)
        results = [
            {
                "content": response["source_document"]["CONCAT_POST_TITLE_DESC"],
                "distance": response["distance"],
                "name": freelancers[str(response["source_document"]["PERSON_ID"])],
            }
            for response in response["responses"]
        ]
        results = [
            {"name": result["name"], "content": result["content"]}
            for result in sorted(results, key=lambda x: x["distance"], reverse=True)
        ]
        return results

    def _make_request(self, payload: dict) -> Results:
        response = requests.post(
            "https://umrsearchservice.umami.staging.platform.usw2.upwork/search/",
            headers=HEADERS,
            json=payload,
            verify=False,
        )
        return response.json()
