from ..retriever import Context, Results, Retriever
from .kg.cypher2neptune import process_cypher_query
from .kg.text2cypher import generate_cypher_query_from_llm
from ..utils import load_prompt
from ..config.constants import SCHEMA
from datetime import date


class KnowledgeGraphRetriever(Retriever):
    RETRIEVER_NAME = "Knowledge Graph"

    def retrieve(self, context: Context) -> Results:
        # Get the prompt from context, or use default
        prompt = context.parameters.get("text2cypher_prompt") or load_prompt(
            "text2cypher.txt"
        )
        prompt = prompt.format(
            schema=SCHEMA,
            messages=context.messages,
            freelancers=context.objects["freelancers"],
            today=date.today(),
            job=context.objects["job"],
        )
        context.parameters["text2cypher_prompt"] = prompt
        generated_cypher = generate_cypher_query_from_llm(context)
        neptune_result = process_cypher_query(generated_cypher)
        return Results(objects=neptune_result, debug={"cypher": generated_cypher})
