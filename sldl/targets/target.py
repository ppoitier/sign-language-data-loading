from abc import ABC, abstractmethod
from typing import Any


class TargetEncoder(ABC):
    """Base class for target encoders used by `SignLanguageDataset`.

    A `TargetEncoder` turns a raw sample dict into a model-ready target
    (e.g. a class index, a per-frame label tensor), and optionally knows how
    to collate a batch of such targets (e.g. padding sequences).
    """

    @abstractmethod
    def encode(self, sample: dict) -> Any:
        """Extract and format the target from a single sample.

        Args:
            sample: The raw sample dict, as produced by
                `SignLanguageDataset` before target encoding.

        Returns:
            The encoded target for this sample.
        """
        pass

    def collate(self, batch_targets: list[Any]) -> Any:
        """Collate a batch of encoded targets.

        Default implementation returns the raw list unchanged. Override this
        for targets that need padding or stacking into a single tensor.

        Args:
            batch_targets: The list of per-sample targets, as returned by
                `encode`, for one batch.

        Returns:
            The collated batch of targets.
        """
        return batch_targets
