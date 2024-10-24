from logging import getLogger
from typing import Generator
from copy import deepcopy
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

DEFAULT_TOP_K_FOR_SINGLE_PERSON = 10

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
        debug: bool = False,
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
        retrievers, person_ids = self._choose_retrievers(context)
        context.objects["target_person_ids"] = person_ids

        if len(person_ids) == 1:
            context.parameters["profile_top_k"] = DEFAULT_TOP_K_FOR_SINGLE_PERSON
            context.parameters["reviews_top_k"] = DEFAULT_TOP_K_FOR_SINGLE_PERSON
            context.parameters["job_history_top_k"] = DEFAULT_TOP_K_FOR_SINGLE_PERSON

        yield "Using the following retrievers:"
        for retriever in retrievers:
            yield f"- {retriever.RETRIEVER_NAME}"

        if not person_ids:
            # Handle retrievers that don't need person_ids
            try:
                for retriever in retrievers:
                    retrieved_data = retriever.retrieve(context)
                    result_text = ""
                    for result_object in retrieved_data.objects:
                        result_object = str(result_object).replace("\\n", "")
                        result_text += f"\n> {result_object}"
                    if debug:
                        result_text += f"\n\nDEBUG: {retrieved_data.debug}"
                    context.objects["results"].append(result_text)
                    yield f"\n## {retriever.RETRIEVER_NAME}"
                    yield result_text
            except Exception as e:
                logger.error(f"Error retrieving data from {retriever}: {e}")
        else:
            # Original person_id based retrieval logic
            retrieval_tasks = itertools.product(retrievers, person_ids)
            results_by_freelancer = {person_id: [] for person_id in person_ids}

            with ThreadPoolExecutor() as executor:
                future_to_task = {
                    executor.submit(self._fetch_data, retriever, context, person_id): (
                        retriever,
                        person_id,
                    )
                    for retriever, person_id in retrieval_tasks
                }
                for future in tqdm(as_completed(future_to_task), total=len(future_to_task)):
                    retriever, person_id = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            results_by_freelancer[person_id].append(
                                (retriever.RETRIEVER_NAME, result)
                            )
                    except Exception as exc:
                        logger.error(
                            f"{retriever.RETRIEVER_NAME} generated an exception: {exc}"
                        )

            for person_id, results in results_by_freelancer.items():
                freelancer_name = next(
                    (
                        freelancer["name"]
                        for freelancer in context.objects["freelancers"]
                        if freelancer["person_id"] == person_id
                    )
                )
                yield f"\n\n# Freelancer {freelancer_name}:"
                for retriever_name, result in results:
                    yield f"\n## {retriever_name}"
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

    def _choose_retrievers(self, context: Context) -> tuple[list[Retriever], list[str]]:
        """
        Chooses the retrievers that are most relevant to the current conversation.

        Args:
            context (Context): The context of the current conversation

        Returns:
            list[Retriever]: The retrievers that are most relevant to the current conversation
        """
        if context.parameters["enforce_rag"] in self.retrievers:
            retrievers, freelancers = (
                [self.retrievers[context.parameters["enforce_rag"]]],
                [
                    freelancer["person_id"]
                    for freelancer in context.objects["freelancers"]
                ],
            )
        elif context.parameters["enforce_rag"] == "Hybrid":
            retrievers, freelancers = (
                self.retrievers.values(),
                [
                    freelancer["person_id"]
                    for freelancer in context.objects["freelancers"]
                ],
            )
        elif context.parameters["enforce_rag"] == "Context-Aware":
            retrievers, freelancers = self.tool_router.choose(context)
        else:
            raise ValueError(f"Invalid RAG value: {context.parameters['enforce_rag']}")
        return retrievers, freelancers

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

    def _fetch_data(self, retriever: Retriever, context: Context, person_id: str) -> str:
        target_freelancers = [
            freelancer
            for freelancer in context.objects["freelancers"]
            if freelancer["person_id"] == person_id
        ]
        retrieve_context = deepcopy(context)
        retrieve_context.objects["freelancers"] = target_freelancers
        try:
            retrieved_data = retriever.retrieve(retrieve_context)
            result_text = ""
            for result_object in retrieved_data.objects:
                result_object = str(result_object).replace("\\n", "")
                result_text += f"\n> {result_object}"
            if context.parameters.get("debug"):
                result_text += f"\n\nDEBUG: {retrieved_data.debug}"
            context.objects["results"].append(result_text)
            return result_text
        except Exception as e:
            logger.error(f"Error retrieving data from {retriever}: {e}")
            return None
