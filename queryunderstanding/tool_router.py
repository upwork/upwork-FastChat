from .retriever import Retriever
from swarm import Swarm, Agent
import json
from logging import getLogger
from .config.constants import RAG_ROUTER_LLM
from .utils import load_prompt

logger = getLogger(__name__)


class ToolRouter:
    def __init__(self, retrievers: dict[str, Retriever]):
        self.swarm = Swarm()
        self.retrievers = retrievers
        self.selector_agent = Agent(
            name="RAG Router",
            model=RAG_ROUTER_LLM,
            instructions=load_prompt("rag_router.txt").format(
                retrievers=", ".join(self.retrievers.keys())
            ),
        )

    def choose(
        self,
        messages: list[dict[str, str]],
    ) -> list[Retriever]:
        response = self.swarm.run(
            agent=self.selector_agent,
            messages=[
                {
                    "role": "user",
                    "content": " ".join([message["content"] for message in messages]),
                },
            ],
        )
        logger.info(f"Retriever Selector Response: {response}")
        try:
            retriever_names = json.loads(response.messages[-1]["content"])
            logger.info(f"Selected retrievers: {retriever_names}")
            return [
                self.retrievers[name]
                for name in retriever_names
                if name in self.retrievers
            ]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error selecting retrievers: {e}")
            return []
