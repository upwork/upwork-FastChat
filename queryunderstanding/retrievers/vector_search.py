from ..retriever import Retriever, Context, Results
from ..data_stores import remote_opensearch
from ..utils import load_prompt, llm_client
from ..config.constants import QUERY_REFORMULATION_LLM


class VectorSearchRetriever(Retriever):
    RETRIEVER_NAME = "Vector Search"

    def __init__(self):
        self.data_store = remote_opensearch.RemoteOpenSearch()

    def retrieve(self, context: Context) -> Results:
        context.objects["query"] = self._reformulate_query(context)
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
