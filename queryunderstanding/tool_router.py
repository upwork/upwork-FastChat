from .retriever import Retriever
from swarm import Swarm, Agent
import json
from logging import getLogger
from .config.constants import RAG_ROUTER_LLM
from .utils import load_prompt
from pydantic import BaseModel
from enum import Enum
from .utils import llm_client

logger = getLogger(__name__)


class Tool(Enum):
    REVIEWS_AND_WORK_HISTORY_SEMANTIC_SEARCH = "Reviews and Work History Semantic Search"
    KNOWLEDGE_GRAPH = "Knowledge Graph"
    HELP_CENTER_SEMANTIC_SEARCH = "Help Center Semantic Search"


class ToolChoice(BaseModel):
    tools: list[Tool]


class ToolRouter:
    def __init__(self, retrievers: dict[str, Retriever]):
        self.retrievers = retrievers
        self.prompt = load_prompt("rag_router.txt").format(
            retrievers=", ".join(self.retrievers.keys())
        )

    def choose(
        self,
        messages: list[dict[str, str]],
    ) -> list[Retriever]:
        response = llm_client.beta.chat.completions.parse(
            model=RAG_ROUTER_LLM,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt,
                },
                {
                    "role": "user",
                    "content": " ".join([message["content"] for message in messages]),
                },
            ],
            response_format=ToolChoice,
        )
        tools = response.choices[0].message.parsed.tools
        tools_names = [tool.value for tool in tools]
        logger.info(f"RAG Router Response: {tools_names}")
        return [
            self.retrievers[name] for name in tools_names if name in self.retrievers
        ]
