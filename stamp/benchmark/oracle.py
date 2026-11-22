"""Scoring interface shared by molecular optimization tasks."""
from abc import ABC, abstractmethod


class Oracle(ABC):
    @abstractmethod
    def __call__(self, smiles: str) -> float:
        """Return a molecule's reward."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the oracle task name."""
