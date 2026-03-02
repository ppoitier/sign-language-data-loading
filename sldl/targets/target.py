from abc import ABC, abstractmethod
from typing import Any


class TargetEncoder(ABC):
    @abstractmethod
    def encode(self, sample: dict) -> Any:
        """
        Extracts and formats the target from a single sample.
        """
        pass

    def collate(self, batch_targets: list[Any]) -> Any:
        """
        Default collation: just return the raw list.
        Override this for tensors that need padding or stacking.
        """
        return batch_targets