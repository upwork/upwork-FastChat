from logging import getLogger
from typing import Generator

from fastchat.conversation import Conversation

from .retriever import Context, Results, Retriever
from .retrievers import knowledge_graph, vector_search
from .data_stores import help_center, reviews_and_work_history, freelancer_profile
from .summarizer import ResultsSummarizer
from .tool_router import ToolRouter
from .utils import load_prompt

logger = getLogger(__name__)

DEFAULT_RETRIEVERS = {
    "Knowledge Graph": knowledge_graph.KnowledgeGraphRetriever(),
    "Reviews and Work History Semantic Search": vector_search.VectorSearchRetriever(
        reviews_and_work_history.ReviewsAndWorkHistorySemanticSearch()
    ),
    "Help Center Semantic Search": vector_search.VectorSearchRetriever(
        help_center.HelpCenterSemanticSearch()
    ),
    "Freelancer Profile Semantic Search": vector_search.VectorSearchRetriever(
        freelancer_profile.FreelancerProfileSemanticSearch()
    ),
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
        text2cypher_prompt: str | None = None,
        query_reformulation_prompt: str | None = None,
        rag_router_prompt: str | None = None,
        enforce_rag_instruction_prompt: str | None = None,
        results_summarizer_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Searches for information relevant to the current conversation.

        Args:
            conversation (Conversation): The current chat's conversation

        Returns:
            list[dict]: The objects retrieved from the data store.
        """
        messages = self._get_messages(conversation)
        breakpoint()
        context = Context(
            messages=messages,
            objects={
                "freelancers": freelancers,
                "job": job,
            },
            parameters={
                "text2cypher_prompt": text2cypher_prompt,
                "query_reformulation_prompt": query_reformulation_prompt,
                "rag_router_prompt": rag_router_prompt,
                "enforce_rag_instruction_prompt": enforce_rag_instruction_prompt,
                "results_summarizer_prompt": results_summarizer_prompt,
                "enforce_rag": enforce_rag,
            },
        )
        retrievers = self._choose_retrievers(context)
        yield "Using the following retrievers:"
        for retriever in retrievers:
            yield f"- {retriever.RETRIEVER_NAME}"
        for retriever in retrievers:
            try:
                retrieved_data: Results = retriever.retrieve(context)
                result_text = f"Retrieved data from {retriever.RETRIEVER_NAME}:\n{retrieved_data}"
                yield result_text
            except Exception as e:
                logger.error(f"Error retrieving data from {retriever}: {e}")
        if summarize_results:
            context.objects["results"] = "\n".join(context.objects.get("results", []))
            summary = self.summarizer.summarize(context)
            yield summary
        job_info = self._get_job_information(context)
        if job_info:
            yield job_info
        freelancer_info = self._get_freelancer_information(context)
        if freelancer_info:
            yield freelancer_info
        instruction = self._enforce_rag_instruction(context)
        yield instruction

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

    def _choose_retrievers(self, context: Context) -> list[Retriever]:
        """
        Chooses the retrievers that are most relevant to the current conversation.

        Args:
            context (Context): The context of the current conversation

        Returns:
            list[Retriever]: The retrievers that are most relevant to the current conversation
        """
        if context.parameters["enforce_rag"] in self.retrievers:
            return [self.retrievers[context.parameters["enforce_rag"]]]
        elif context.parameters["enforce_rag"] == "Hybrid":
            return self.retrievers.values()
        elif context.parameters["enforce_rag"] == "Context-Aware":
            return self.tool_router.choose(context)
        else:
            raise ValueError(f"Invalid RAG value: {context.parameters['enforce_rag']}")

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

    def _get_job_information(self, context: Context) -> str:
        """
        Gets the information of the job.
        """
        job = context.objects.get("job")
        if not job:
            return ""
        return (
            f"\n\n### Job Information\n\n"
            f"Title: {job['title']}\n\n"
            f"Description: {job['description']}"
        )

    def _get_freelancer_information(self, context: Context) -> str:
        """
        Gets the information of the freelancers.
        """
        freelancers = context.objects.get("freelancers")
        if not freelancers:
            return ""
        freelancer_info = []
        for freelancer in freelancers:
            freelancer_info.append(f"Name: {freelancer['name']}\n")
            freelancer_info.append(f"Title: {freelancer['title']}\n")
        return f"\n\n### Freelancer Information\n\nFreelancers:\n{freelancer_info}"

    def _enforce_rag_instruction(self, context: Context) -> str:
        """
        Enforces the RAG instruction.
        """
        prompt = context.parameters["enforce_rag_instruction_prompt"]
        return "\n\n" + prompt
