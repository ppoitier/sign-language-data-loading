from typing import Any

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    raise ImportError(
        f"PyTorch is not installed. Please install it to use example targets."
    )

from sldl.targets import TargetEncoder


class SegmentTarget(TargetEncoder):
    def __init__(self, annotation_id: str = 'both_hands'):
        super().__init__()
        self.annotation_id = annotation_id

    def encode(self, sample: dict) -> Any:
        annots = sample['annotations'][self.annotation_id]
        return torch.from_numpy(annots.loc[:, ['start_frame', 'end_frame']].values).long()
