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
    from sign_language_tools.annotations.transforms import SegmentsToFrameLabels
except ImportError:
    raise ImportError(
        f"sign_language_tools is not installed. Please install it to use example targets."
    )


class FrameLabelsTarget(TargetEncoder):
    """Renders annotations as a per-frame label tensor.

    Supports binary (active/background) and categorical (class IDs) modes
    depending on whether `label_to_id` is provided. An optional
    `segment_transform` is applied to the (N, 2) segments array before
    rendering — use this to plug in RemoveOverlapping or any other
    segment-level cleanup.
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        column: str = "lemma",
        label_to_id: dict[str, int] | None = None,
        background_id: int = 0,
        active_id: int = 1,
        unknown_id: int = -1,
        unknown_label: str = "<unk>",
        pad_value: int = -100,
        segment_transform: Callable | None = None,
    ):
        self.annotation_id = annotation_id
        self.column = column
        self.label_to_id = label_to_id
        self.background_id = background_id
        self.unknown_id = unknown_id
        self.unknown_label = unknown_label
        self.pad_value = pad_value
        self.segment_transform = segment_transform
        self._to_frame_labels = SegmentsToFrameLabels(
            background_label=background_id,
            fill_label=active_id,
        )

    def encode(self, sample: dict) -> Any:
        n_frames = sample.get("n_frames", 0)
        annotations = sample.get("annotations", {}).get(self.annotation_id)

        if annotations is None or annotations.empty:
            return torch.from_numpy(
                np.full(n_frames, self.background_id, dtype=np.int64)
            )

        starts = annotations["start_frame"].to_numpy()
        ends = annotations["end_frame"].to_numpy()

        if self.label_to_id is None:
            segments = np.stack([starts, ends], axis=1)
        else:
            raw = annotations[self.column].fillna(self.unknown_label)
            values = raw.map(lambda l: self.label_to_id.get(l, self.unknown_id)).to_numpy()
            segments = np.stack([starts, ends, values], axis=1)

        if self.segment_transform is not None:
            segments = self.segment_transform(segments)

        labels = self._to_frame_labels(segments, vector_size=n_frames)
        return torch.from_numpy(labels)

    def collate(self, batch_targets: list[Any]) -> Any:
        return pad_sequence(batch_targets, batch_first=True, padding_value=self.pad_value)
