from typing import Any

from sldl.targets.target import TargetEncoder
from sldl.utils.json import from_json_to_dict

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    raise ImportError(
        f"PyTorch is not installed. Please install it to use example targets."
    )


class ContinuousRecognitionTarget(TargetEncoder):
    """Extracts a temporal sequence of labels for continuous recognition tasks.

    Args:
        annotation_id: Key of the annotation track to read from
            `sample["annotations"]`.
        column: Name of the label column to extract from the annotation
            track.
        label_to_id: Mapping from label string to class id, or a path to a
            JSON file containing such a mapping. If `None`, `encode` returns
            the raw label strings instead of ids.
        unknown_id: Class id assigned to labels missing from `label_to_id`.
        unknown_label: Placeholder used for missing/NaN label values.
        pad_value: Padding value used when collating a batch (only used
            when `label_to_id` is set).
        collapse_adjacent_duplicates: If True, consecutive repeated labels
            (e.g. two annotation rows with the same lemma back-to-back) are
            merged into a single occurrence. CTC needs a blank between two
            adjacent-identical target labels to keep them distinguishable
            after decoding's repeat-collapsing step, so a raw sequence with
            adjacent duplicates silently raises the input length required
            for that window to be CTC-feasible. Off by default to preserve
            the exact annotated sequence for consumers that don't train
            with CTC.
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        column: str = "lemma",
        label_to_id: dict[str, int] | str | None = None,
        unknown_id: int = -1,
        unknown_label: str = '<unk>',
        pad_value: int = 0,
        collapse_adjacent_duplicates: bool = False,
    ):
        self.annotation_id = annotation_id
        self.column = column
        self.label_to_id = label_to_id
        if isinstance(self.label_to_id, str):
            self.label_to_id = from_json_to_dict(self.label_to_id)
        self.unknown_id = unknown_id
        self.unknown_label = unknown_label
        self.pad_value = pad_value
        self.collapse_adjacent_duplicates = collapse_adjacent_duplicates

    def encode(self, sample: dict) -> Any:
        """Extract the label sequence for a single sample.

        Args:
            sample: The raw sample dict, expected to contain an
                `sample["annotations"][self.annotation_id]` DataFrame.

        Returns:
            A list of label ids (if `label_to_id` is set) or raw label
            strings, in temporal order. Empty if there are no annotations.
        """
        annotations = sample.get("annotations", {}).get(self.annotation_id)

        if annotations is None or annotations.empty:
            sequence = []
        else:
            sequence = annotations[self.column].fillna(self.unknown_label).tolist()

        if self.collapse_adjacent_duplicates:
            sequence = [label for i, label in enumerate(sequence) if i == 0 or label != sequence[i - 1]]

        if self.label_to_id is not None:
            if self.unknown_id < 0:
                # A negative unknown_id can never be a valid downstream class id (e.g.
                # embedding lookups, F.ctc_loss targets), so silently falling back to it
                # doesn't fail until it corrupts memory on the GPU. Fail loudly here instead.
                for label in sequence:
                    if label not in self.label_to_id:
                        raise KeyError(
                            f"Label {label!r} is missing from label_to_id "
                            f"(annotation_id={self.annotation_id!r}, column={self.column!r}), "
                            f"and unknown_id={self.unknown_id} is negative -- not a valid class id. "
                            "Add the label to the vocabulary, or pass a valid non-negative "
                            "unknown_id if out-of-vocabulary labels are expected."
                        )
            return [self.label_to_id.get(label, self.unknown_id) for label in sequence]

        return sequence

    def collate(self, batch_targets: list[Any]) -> Any:
        """Pad a batch of label-id sequences to the same length.

        Args:
            batch_targets: The list of per-sample label sequences, as
                returned by `encode`.

        Returns:
            A `(batch, max_len)` `torch.long` tensor padded with
            `self.pad_value` if `label_to_id` is set, otherwise the raw
            `batch_targets` list unchanged.
        """
        if self.label_to_id is not None:
            tensors = [torch.tensor(seq, dtype=torch.long) for seq in batch_targets]
            return pad_sequence(tensors, batch_first=True, padding_value=self.pad_value)

        return super().collate(batch_targets)
