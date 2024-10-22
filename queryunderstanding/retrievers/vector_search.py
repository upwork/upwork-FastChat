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
        return self.data_store.search(context)

    def _reformulate_query(self, context: Context) -> str:
        full_conversation = " ".join(
            [message["content"] for message in context.messages]
        )
        if context.objects["job"]:
            full_conversation += (
                f"\n\n### Job Information\n\n"
                f"Title: {context.objects['job']['title']}\n\n"
                f"Description: {context.objects['job']['description']}"
            )
        response = llm_client.chat.completions.create(
            model=QUERY_REFORMULATION_LLM,
            messages=[
                {"role": "system", "content": load_prompt("query_reformulation.txt")},
                {"role": "user", "content": full_conversation},
            ],
        )
        return response.choices[0].message.content
