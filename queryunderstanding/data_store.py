from abc import ABC, abstractmethod


class DataStore(ABC):
    @abstractmethod
    def search(
        self,
        query: str,  # KG = Cypher; Vector Search = NL Query
    ) -> list[dict]:
        """
        Performs a search in the data store.

        Args:
            query (str): The query to perform. Can be a Cypher query (KG) or a NL query (Vector Search).
            Examples:
                Cypher Query: "MATCH (n:Person) WHERE n.name = 'John Doe' RETURN n"
                NL Query: "Python Developer"

        Returns:
            list[dict]: The objects retrieved from the data store.
        """
        pass
