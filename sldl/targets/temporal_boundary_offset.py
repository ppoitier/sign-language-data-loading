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


class TemporalBoundaryOffsetTarget(TargetEncoder):
    """
    Creates a time-series where each active frame represents its temporal
    distance to the start or end of the current sign.
    """

    def __init__(
            self,
            annotation_id: str = "both_hands",
            ref_location: Literal["start", "end"] = "start",
            background_value: float = -1.0,
            pad_value: float = -1.0,
    ):
        self.annotation_id = annotation_id
        if ref_location not in ["start", "end"]:
            raise ValueError(f"ref_location must be 'start' or 'end', got {ref_location}")
        self.ref_location = ref_location
        self.background_value = background_value
        self.pad_value = pad_value

    def encode(self, sample: dict) -> Any:
        n_frames = sample.get("n_frames", 0)
        offset_series = np.full(n_frames, self.background_value, dtype=np.float32)

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
                if self.ref_location == "start":
                    offset_series[slice_idx] = time_indices[slice_idx] - start
                elif self.ref_location == "end":
                    offset_series[slice_idx] = end - time_indices[slice_idx]

        return torch.from_numpy(offset_series)

    def collate(self, batch_targets: list[Any]) -> Any:
        return pad_sequence(batch_targets, batch_first=True, padding_value=self.pad_value)