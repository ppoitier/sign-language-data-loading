from typing import Any, Literal
import numpy as np

from sldl.targets.target import TargetEncoder

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    raise ImportError(
        f"PyTorch is not installed. Please install it to use example targets."
    )


class TemporalBoundaryOffsetsTarget(TargetEncoder):
    """
    Creates a time-series where each active frame represents its temporal
    distance to the start and end of the current sign.
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        background_value: float = -1.0,
        pad_value: float = -1.0,
    ):
        self.annotation_id = annotation_id
        self.background_value = background_value
        self.pad_value = pad_value

    def encode(self, sample: dict) -> Any:
        n_frames = sample.get("n_frames", 0)
        offset_series = np.full((n_frames, 2), self.background_value, dtype=np.float32)

        annotations = sample.get("annotations", {}).get(self.annotation_id)
        if annotations is not None and not annotations.empty:
            time_indices = np.arange(n_frames)

            for _, row in annotations.iterrows():
                start = int(row["start_frame"])
                end = int(row["end_frame"])
                start_clip = max(0, start)
                end_clip = min(n_frames, end)

                if start_clip >= end_clip:
                    continue

                slice_idx = slice(start_clip, end_clip)
                offset_series[slice_idx, 0] = time_indices[slice_idx] - start
                offset_series[slice_idx, 1] = end - time_indices[slice_idx]

        return torch.from_numpy(offset_series)

    def collate(self, batch_targets: list[Any]) -> Any:
        return pad_sequence(
            batch_targets, batch_first=True, padding_value=self.pad_value
        )
