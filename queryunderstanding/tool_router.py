from .retriever import Retriever, Context
from swarm import Swarm, Agent
import json
from logging import getLogger

logger = getLogger(__name__)


class ToolRouter:
    def __init__(self, retrievers: dict[str, Retriever]):
        self.swarm = Swarm()
        self.retrievers = retrievers
        self.selector_agent = Agent(
            name="Retriever Selector",
            instructions=f"""
            You are a helpful assistant that decides which retriever(s) to use based on the conversation.
            Available retrievers are: {', '.join(self.retrievers.keys())}.
            Return the retriever name(s) as a JSON list.

            The vector search indices contain information about the freelancer profile, freelancer work history and past projects feedbacks.
            The knowledge graph contains information about Upwork's database of freelancers, jobs, clients, skills, etc.

            Examples:
            - How many active contracts does X have? -> ["Knowledge Graph"]
            - How many of the person's previous jobs needed GenAI skills? -> ["Knowledge Graph"]
            - What database work experience does this person have? -> ["Vector Search", "Knowledge Graph"]
            - What do customers have to say about their communication skills? -> ["Vector Search"]
            - "What are the top 3 categories at Upwork?" -> ["Knowledge Graph"]
            - "Does this freelancer know about AI?" -> ["Vector Search"]
            - "List the pros and cons of this freelancer's profile." -> ["Vector Search"]
            - "How many jobs have this freelancer done in the last 3 months?" -> ["Knowledge Graph"]
            - "What were the top 3 projects this freelancer worked on?" -> ["Vector Search", "Knowledge Graph"]
""",
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
