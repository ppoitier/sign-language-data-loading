from typing import Any, Callable

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    raise ImportError(
        f"PyTorch is not installed. Please install it to use example targets."
    )

from sldl.targets import TargetEncoder


class SegmentTarget(TargetEncoder):
    def __init__(
        self,
        annotation_id: str = "both_hands",
        segment_transform: Callable | None = None,
    ):
        super().__init__()
        self.annotation_id = annotation_id
        self.segment_transform = segment_transform

    def encode(self, sample: dict) -> Any:
        annots = sample["annotations"][self.annotation_id]
        segments = annots.loc[:, ["start_frame", "end_frame"]].values
        if self.segment_transform is not None:
            segments = self.segment_transform(segments)
        return torch.from_numpy(segments).long()
