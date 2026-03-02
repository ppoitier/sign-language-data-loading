from typing import Any

from sldl.targets.target import TargetEncoder

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    raise ImportError(
        f"PyTorch is not installed. Please install it to use example targets."
    )


class ContinuousRecognitionTarget(TargetEncoder):
    """Extracts a temporal sequence of labels for continuous recognition tasks."""

    def __init__(
        self,
        annotation_id: str = "both_hands",
        column: str = "lemma",
        label_to_id: dict[str, int] | None = None,
        unknown_id: int = -1,
        unknown_label: str = '<unk>',
        pad_value: int = 0,
    ):
        self.annotation_id = annotation_id
        self.column = column
        self.label_to_id = label_to_id
        self.unknown_id = unknown_id
        self.unknown_label = unknown_label
        self.pad_value = pad_value

    def encode(self, sample: dict) -> Any:
        annotations = sample.get("annotations", {}).get(self.annotation_id)

        if annotations is None or annotations.empty:
            sequence = []
        else:
            sequence = annotations[self.column].fillna(self.unknown_label).tolist()

        if self.label_to_id is not None:
            return [self.label_to_id.get(label, self.unknown_id) for label in sequence]

        return sequence

    def collate(self, batch_targets: list[Any]) -> Any:
        if self.label_to_id is not None:
            tensors = [torch.tensor(seq, dtype=torch.long) for seq in batch_targets]
            return pad_sequence(tensors, batch_first=True, padding_value=self.pad_value)

        return super().collate(batch_targets)
