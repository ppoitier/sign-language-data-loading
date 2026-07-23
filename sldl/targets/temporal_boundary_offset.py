from typing import Any, Callable

import numpy as np

from sldl.targets.target import TargetEncoder

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    raise ImportError(
        f"PyTorch is not installed. Please install it to use example targets."
    )

try:
    from sign_language_tools.annotations.transforms import (
        SegmentsToBoundaryOffsets,
    )
except ImportError:
    raise ImportError(
        f"sign_language_tools is not installed. Please install it to use example targets."
    )


class TemporalBoundaryOffsetsTarget(TargetEncoder):
    """Per-frame regression target: distance to current segment's boundaries.

    Output shape (after collate): (batch, 2, time), where channel 0 is
    `start_offset` and channel 1 is `end_offset`. Background frames carry
    `pad_value` in both channels.

    Args:
        annotation_id: Key of the annotation track to read from
            `sample["annotations"]`.
        background_value: Offset value assigned to background frames (frames
            outside any segment).
        pad_value: Padding value used when collating a batch.
        segment_transform: A callable applied to the `(N, 2)` segments array
            before rendering.
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        background_value: float = -1.0,
        pad_value: float = -1.0,
        segment_transform: Callable | None = None,
    ):
        super().__init__()
        self.annotation_id = annotation_id
        self.background_value = background_value
        self.pad_value = pad_value
        self.segment_transform = segment_transform
        self._renderer = SegmentsToBoundaryOffsets(
            background_value=background_value,
        )

    def encode(self, sample: dict) -> Any:
        """Render the per-frame boundary-offset tensor for a single sample.

        Args:
            sample: The raw sample dict. Must contain `n_frames` and, unless
                the annotation track is missing/empty, an
                `sample["annotations"][self.annotation_id]` DataFrame with
                `start_frame` / `end_frame` columns.

        Returns:
            A `(n_frames, 2)` `torch.float32` tensor of
            `(start_offset, end_offset)` pairs.
        """
        n_frames = sample.get("n_frames", 0)
        annotations = sample.get("annotations", {}).get(self.annotation_id)

        if annotations is None or annotations.empty:
            empty = np.full((n_frames, 2), self.background_value, dtype=np.float32)
            return torch.from_numpy(empty)

        segments = annotations[["start_frame", "end_frame"]].to_numpy()
        if self.segment_transform is not None:
            segments = self.segment_transform(segments)

        offsets = self._renderer(segments, sequence_length=n_frames)
        return torch.from_numpy(offsets)

    def collate(self, batch_targets: list[Any]) -> Any:
        """Pad and stack a batch of boundary-offset tensors.

        Args:
            batch_targets: The list of per-sample `(n_frames, 2)` tensors,
                as returned by `encode`.

        Returns:
            A `(batch, 2, max_n_frames)` tensor, padded with
            `self.pad_value`.
        """
        # pad_sequence stacks along time axis: (batch, time, 2).
        # Permute to channel-first (batch, 2, time) for sequence models.
        return pad_sequence(
            batch_targets, batch_first=True, padding_value=self.pad_value
        ).permute(0, 2, 1)
