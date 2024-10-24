from logging import getLogger

from ..config.constants import QUERY_REFORMULATION_LLM
from ..retriever import Context, Results, Retriever
from ..utils import llm_client, load_prompt

logger = getLogger(__name__)


class VectorSearchRetriever(Retriever):
    RETRIEVER_NAME = "Vector Search"

    def __init__(self, data_store):
        super().__init__(data_store)
        self.RETRIEVER_NAME = (
            self.RETRIEVER_NAME + " - " + self.data_store.DATA_STORE_NAME
        )

    def retrieve(self, context: Context) -> Results:
        context.objects["query"] = self._reformulate_query(context)
        logger.info(f"Vector Search Query: {context.objects['query']}")
        results = self.data_store.search(context)
        results.debug = {"query": context.objects["query"]}
        return results

    def _reformulate_query(self, context: Context) -> str:
        full_conversation = " ".join(
            [message["content"] for message in context.messages]
        )
        prompt = context.parameters.get("query_reformulation_prompt") or load_prompt(
            "query_reformulation.txt"
        )

        response = llm_client.chat.completions.create(
            model=QUERY_REFORMULATION_LLM,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": full_conversation},
            ],
        )
        return response.choices[0].message.content
