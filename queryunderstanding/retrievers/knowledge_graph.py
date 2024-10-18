from ..retriever import Context, Results, Retriever
from .kg.cypher2neptune import process_cypher_query
from .kg.text2cypher import generate_cypher_query_from_llm


class KnowledgeGraphRetriever(Retriever):
    RETRIEVER_NAME = "Knowledge Graph"

    def __init__(self):
        self.__name__ = "Knowledge Graph"

    def retrieve(self, context: Context) -> Results:
        generated_cypher = generate_cypher_query_from_llm(context)
        neptune_result = process_cypher_query(generated_cypher)
        return Results(objects=neptune_result)
