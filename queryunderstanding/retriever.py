from abc import ABC, abstractmethod
from dataclasses import dataclass

from .data_store import DataStore


@dataclass
class Context:
    messages: list[dict]
    objects: dict  # Query, Freelancer Id, Job, etc.


@dataclass
class Results:
    objects: list[dict]


class Retriever(ABC):
    def __init__(self, data_store: DataStore = None):
        self.data_store = data_store

    @abstractmethod
    def retrieve(self, messages: Context) -> Results:
        """
        Retrieves information from the data store.

        Args:
            messages (Context): The context of the current conversation

        Returns:
            Results: The objects retrieved from the data store.
        """
        pass
