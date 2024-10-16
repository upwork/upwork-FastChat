from retriever import Retriever


class QueryUnderstanding:
    def __init__(self, retrievers: list[Retriever]):
        self.retrievers = retrievers

    def search(self, messages: list[dict[str, str]]) -> list[dict]:
        """
        Searches for information relevant to the current conversation.

        Args:
            messages (list[dict[str, str]]): The messages of the current conversation

        Returns:
            list[dict]: The objects retrieved from the data store.
        """
        retrievers = self._choose_retrievers(messages)
        results = []
        for retriever in retrievers:
            results.extend(retriever.retrieve(messages))
        return results

    def _choose_retrievers(self, messages: list[dict[str, str]]) -> list[Retriever]:
        """
        Chooses the retrievers that are most relevant to the current conversation.

        Args:
            messages (list[dict[str, str]]): The messages of the current conversation

        Returns:
            list[Retriever]: The retrievers that are most relevant to the current conversation
        """
        return self.retrievers
