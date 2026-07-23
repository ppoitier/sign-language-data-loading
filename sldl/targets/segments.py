from typing import Any, Callable

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    raise ImportError(
        f"PyTorch is not installed. Please install it to use example targets."
    )

from sldl.targets.target import TargetEncoder


class SegmentTarget(TargetEncoder):
    """Extracts raw (start_frame, end_frame) segments from an annotation track.

    Args:
        annotation_id: Key of the annotation track to read from
            `sample["annotations"]`.
        segment_transform: A callable applied to the `(N, 2)` segments array
            (e.g. `RemoveOverlapping`) before it is converted to a tensor.
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        segment_transform: Callable | None = None,
    ):
        super().__init__()
        self.annotation_id = annotation_id
        self.segment_transform = segment_transform

    def encode(self, sample: dict) -> Any:
        """Extract the segments for a single sample.

        Args:
            sample: The raw sample dict. Must contain an
                `sample["annotations"][self.annotation_id]` DataFrame with
                `start_frame` / `end_frame` columns.

        Returns:
            A `(N, 2)` `torch.long` tensor of `(start_frame, end_frame)` pairs.
        """
        annots = sample["annotations"][self.annotation_id]
        segments = annots.loc[:, ["start_frame", "end_frame"]].values
        if self.segment_transform is not None:
            segments = self.segment_transform(segments)
        return torch.from_numpy(segments).long()
