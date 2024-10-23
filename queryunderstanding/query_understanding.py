from logging import getLogger
from typing import Generator

from fastchat.conversation import Conversation

from .retriever import Context, Results, Retriever
from .retrievers import knowledge_graph, vector_search
from .data_stores import help_center, reviews_and_work_history, freelancer_profile
from .summarizer import ResultsSummarizer
from .tool_router import ToolRouter

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
        help_center_top_k: int = 10,
        job_history_top_k: int = 10,
        reviews_top_k: int = 10,
        profile_top_k: int = 10,
    ) -> Generator[str, None, None]:
        """
        Searches for information relevant to the current conversation.

        Args:
            conversation (Conversation): The current chat's conversation

        Returns:
            list[dict]: The objects retrieved from the data store.
        """
        messages = self._get_messages(conversation)
        context = Context(
            messages=messages,
            objects={
                "freelancers": freelancers,
                "job": job,
                "results": [],
            },
            parameters={
                "text2cypher_prompt": text2cypher_prompt,
                "query_reformulation_prompt": query_reformulation_prompt,
                "rag_router_prompt": rag_router_prompt,
                "enforce_rag_instruction_prompt": enforce_rag_instruction_prompt,
                "results_summarizer_prompt": results_summarizer_prompt,
                "enforce_rag": enforce_rag,
                "help_center_top_k": help_center_top_k,
                "job_history_top_k": job_history_top_k,
                "reviews_top_k": reviews_top_k,
                "profile_top_k": profile_top_k,
            },
        )
        retrievers = self._choose_retrievers(context)
        yield "Using the following retrievers:"
        for retriever in retrievers:
            yield f"- {retriever.RETRIEVER_NAME}"
        import asyncio
        import concurrent.futures

        async def fetch_data(retriever, context):
            try:
                retrieved_data: Results = await asyncio.to_thread(
                    retriever.retrieve, context
                )
                result_text = f"\n\n\nRetrieved data from {retriever.RETRIEVER_NAME}:\n"
                for result_object in retrieved_data.objects:
                    result_object = str(result_object).replace("\\n", "")
                    result_text += f"\n*   | {result_object}"
                result_text += f"\n\n*   | {retrieved_data.debug}"
                context.objects["results"].append(result_text)
                if not summarize_results:
                    return result_text
            except Exception as e:
                logger.error(f"Error retrieving data from {retriever}: {e}")
                return None

        async def run_tasks(retrievers, context):
            tasks = [fetch_data(retriever, context) for retriever in retrievers]
            return await asyncio.gather(*tasks)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_tasks(retrievers, context))
            loop.close()

        for result in results:
            if result:
                yield result
        if summarize_results:
            summary = self.summarizer.summarize(context)
            yield summary
        job_info = self._get_job_information(context)
        if job_info:
            yield job_info
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
        for message in conversation.messages:
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

    def _enforce_rag_instruction(self, context: Context) -> str:
        """
        Enforces the RAG instruction.
        """
        prompt = context.parameters["enforce_rag_instruction_prompt"]
        return "\n\n" + prompt
