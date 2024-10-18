from ..retriever import Retriever, Context, Results
from ..data_stores import remote_opensearch
from ..utils import load_prompt, llm_client
from ..config.constants import QUERY_REFORMULATION_LLM



class VectorSearchRetriever(Retriever):
    RETRIEVER_NAME = "Vector Search"

    def __init__(self):
        self.data_store = remote_opensearch.RemoteOpenSearch()

    def retrieve(self, context: Context) -> Results:
        context.objects["query"] = self._reformulate_query(context.messages)
        return self.data_store.search(context)

    def _reformulate_query(self, messages: list[dict[str, str]]) -> str:
        full_conversation = " ".join([message["content"] for message in messages])
        response = llm_client.chat.completions.create(
            model=QUERY_REFORMULATION_LLM,
            messages=[
                {
                    "role": "system",
                    "content": load_prompt("query_reformulation.txt")
                },
                {"role": "user", "content": full_conversation},
            ],
        )
        return response.choices[0].message.content
