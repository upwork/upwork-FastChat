from enum import Enum
from logging import getLogger

from pydantic import BaseModel

from .config.constants import RAG_ROUTER_LLM
from .retriever import Context, Retriever
from .utils import llm_client

logger = getLogger(__name__)


class Tool(Enum):
    REVIEWS_AND_WORK_HISTORY_SEMANTIC_SEARCH = (
        "Reviews and Work History Semantic Search"
    )
    KNOWLEDGE_GRAPH = "Knowledge Graph"
    HELP_CENTER_SEMANTIC_SEARCH = "Help Center Semantic Search"
    FREELANCER_PROFILE_SEMANTIC_SEARCH = "Freelancer Profile Semantic Search"


class ToolChoice(BaseModel):
    tools: list[Tool]
    person_ids: list[str]


class ToolRouter:
    def __init__(self, retrievers: dict[str, Retriever]):
        self.retrievers = retrievers

    def choose(self, context: Context) -> list[Retriever]:
        prompt = context.parameters["rag_router_prompt"]
        prompt = prompt.format(
            retrievers=", ".join(self.retrievers.keys()),
            freelancers="\n".join(
                [
                    f"- Person ID: {freelancer['person_id']}, Name: {freelancer['name']}"
                    for freelancer in context.objects["freelancers"]
                ]
            ),
        )
        response = llm_client.beta.chat.completions.parse(
            model=RAG_ROUTER_LLM,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": " ".join(
                        [message["content"] for message in context.messages]
                    ),
                },
            ],
            response_format=ToolChoice,
        )
        tools = response.choices[0].message.parsed.tools
        tools_names = [tool.value for tool in tools]
        tools_names = self._apply_rules(context, tools_names)
        logger.info(f"RAG Router Response: {tools_names}")
        retrievers = [
            self.retrievers[name] for name in tools_names if name in self.retrievers
        ]
        return retrievers, response.choices[0].message.parsed.person_ids

    def _apply_rules(self, context: Context, tools_names: list[str]) -> list[str]:
        messages = context.messages
        last_message = messages[-1]["content"]
        if "skills" in last_message.lower():
            if not any(tool == Tool.KNOWLEDGE_GRAPH.value for tool in tools_names):
                tools_names.append(Tool.KNOWLEDGE_GRAPH.value)
        if "hourly rate" in last_message.lower():
            if not any(tool == Tool.KNOWLEDGE_GRAPH.value for tool in tools_names):
                tools_names.append(Tool.KNOWLEDGE_GRAPH.value)
        if "success score" in last_message.lower():
            if not any(tool == Tool.KNOWLEDGE_GRAPH.value for tool in tools_names):
                tools_names.append(Tool.KNOWLEDGE_GRAPH.value)
        if "contract" in last_message.lower():
            if not any(tool == Tool.KNOWLEDGE_GRAPH.value for tool in tools_names):
                tools_names.append(Tool.KNOWLEDGE_GRAPH.value)
        return tools_names
