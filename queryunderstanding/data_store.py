from abc import ABC, abstractmethod


class DataStore(ABC):
    def __init__(self, config: dict):
        self.client = self.connect()

    @abstractmethod
    def connect(self):
        """
        Connects to the data store.
        This can be a local endpoint or a remote endpoint, as an example.
        """
        pass

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
