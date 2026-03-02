from typing import Any
import numpy as np

from sldl.targets.target import TargetEncoder

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    raise ImportError(
        f"PyTorch is not installed. Please install it to use example targets."
    )


class TemporalSegmentationTarget(TargetEncoder):
    """
    Creates a frame-level temporal segmentation mask.
    Supports both binary masks (active/inactive) and categorical masks (label IDs).
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        column: str = "lemma",
        label_to_id: dict[str, int] | None = None,
        background_id: int = 0,
        active_id: int = 1,      # Used if label_to_id is None (Binary mode)
        unknown_id: int = -1,    # Used if label_to_id is provided but label is missing
        unknown_label: str = "<unk>",
        pad_value: int = -100,      # Usually the background_id or a special ignore_index (like -100)
    ):
        self.annotation_id = annotation_id
        self.column = column
        self.label_to_id = label_to_id
        self.background_id = background_id
        self.active_id = active_id
        self.unknown_id = unknown_id
        self.unknown_label = unknown_label
        self.pad_value = pad_value

    def encode(self, sample: dict) -> Any:
        n_frames = sample.get("n_frames", 0)
        mask = np.full(n_frames, self.background_id, dtype=np.int64)
        annotations = sample.get("annotations", {}).get(self.annotation_id)
        if annotations is not None and not annotations.empty:
            for _, row in annotations.iterrows():
                start = max(0, int(row["start_frame"]))
                end = min(n_frames, int(row["end_frame"]))

                if self.label_to_id is None:
                    val = self.active_id
                else:
                    label = row[self.column]
                    if label is None:
                        label = self.unknown_label
                    val = self.label_to_id.get(label, self.unknown_id)
                mask[start:end] = val

        return torch.from_numpy(mask)

    def collate(self, batch_targets: list[Any]) -> Any:
        return pad_sequence(batch_targets, batch_first=True, padding_value=self.pad_value)