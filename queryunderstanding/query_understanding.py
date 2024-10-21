import json
from logging import getLogger

from fastchat.conversation import Conversation

from .retriever import Context, Results, Retriever
from .retrievers import knowledge_graph, vector_search
from .summarizer import ResultsSummarizer
from .tool_router import ToolRouter

logger = getLogger(__name__)

DEFAULT_RETRIEVERS = {
    "Knowledge Graph": knowledge_graph.KnowledgeGraphRetriever(),
    "Vector Search": vector_search.VectorSearchRetriever(),
}


class QueryUnderstanding:
    def __init__(self):
        self.retrievers = DEFAULT_RETRIEVERS
        self.summarizer = ResultsSummarizer()
        self.tool_router = ToolRouter(self.retrievers)

    def search(
        self,
        conversation: Conversation,
        summarize_results: bool,
        freelancers: list[dict[str, str]],
        job: dict[str, str],
        enforce_rag: str | None = None,
    ) -> str:
        """
        Searches for information relevant to the current conversation.

        Args:
            conversation (Conversation): The current chat's conversation

        Returns:
            list[dict]: The objects retrieved from the data store.
        """
        messages: list[dict[str, str]] = self._get_messages(conversation)
        logger.info(f"Query Understanding Messages: {messages}")
        retrievers = self._choose_retrievers(messages, enforce_rag)
        context = Context(
            messages=messages,
            objects={
                "freelancers": freelancers,
                "job": job,
            },
        )
        results = [
            f">>>>>>>>> Using the following retrievers: {[retriever.RETRIEVER_NAME for retriever in retrievers]} <<<<<<<",
        ]
        for retriever in retrievers:
            try:
                results.append(self._fetch_data(retriever, context))
            except Exception as e:
                logger.error(f"Error retrieving data from {retriever}: {e}")
        result_text = "\n".join(results)
        result_text += (
            f"\n\n### Job Information\n\n"
            f"Title: {job['title']}\n\n"
            f"Description: {job['description']}"
        )
        if summarize_results:
            context.objects["results"] = result_text
            result_text = self.summarizer.summarize(context)
        return result_text

    def _get_messages(self, conversation: Conversation) -> list[dict]:
        """
        Gets the messages of the current conversation.

        Args:
            conversation (Conversation): The current chat's conversation

        Returns:
            list[dict[str, str]]: The messages of the current conversation
        """
        messages = []
        for message in conversation.messages[:-1]:
            role = message[0]
            text = message[1]
            messages.append({"role": role, "content": text})
        return messages

    def _choose_retrievers(
        self, messages: list[dict[str, str]], enforce_rag: str | None = None
    ) -> list[Retriever]:
        """
        Chooses the retrievers that are most relevant to the current conversation.

        Args:
            messages (list[dict[str, str]]): The messages of the current conversation

        Returns:
            list[Retriever]: The retrievers that are most relevant to the current conversation
        """
        if enforce_rag in self.retrievers:
            return [self.retrievers[enforce_rag]]
        elif enforce_rag == "Hybrid":
            return self.retrievers.values()
        elif enforce_rag == "Context-Aware":
            return self.tool_router.choose(messages)
        else:
            raise ValueError(f"Invalid RAG value: {enforce_rag}")

    def _fetch_data(self, retriever: Retriever, context: Context) -> str:
        """
        Fetches the data from the retriever.

        Args:
            retriever (Retriever): The retriever to fetch data from
            context (Context): The context of the current conversation

        Returns:
            str: The formatted retrieved data
        """
        retrieved_data: Results = retriever.retrieve(context)
        return f"""
        Retrieved data from {retriever.RETRIEVER_NAME}:
        {retrieved_data}
        """
