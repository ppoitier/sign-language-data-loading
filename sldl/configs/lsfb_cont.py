from typing import Literal

from pydantic import Field, model_validator

from sldl.config import SignLanguageDatasetConfig
from sldl.targets.frame_labels import FrameLabelsTarget
from sldl.targets.target import TargetEncoder

LSFBContSplit = Literal["train", "eval", "test"]

_BODY_PARTS = (
    "upper_pose",
    "left_hand",
    "right_hand",
    "lips",
    "left_eye",
    "right_eye",
)


class LSFBContConfig(SignLanguageDatasetConfig):
    """Config preset for the LSFB-CONT continuous sign dataset.

    `shards_url`, `video_path` and `video_index_path` are derived from
    `root` / `split` unless explicitly overridden.

    Example::

        config = LSFBContConfig(
            root="F:/datasets/sign-language/lsfb-cont",
            split="train",
        )
        ds = SignLanguageDataset.from_config(config)
    """

    root: str
    split: LSFBContSplit

    body_parts: tuple[str, ...] = _BODY_PARTS
    show_loading_progress: bool = True
    load_videos: bool = True
    targets: dict[str, TargetEncoder] | None = Field(
        default_factory=lambda: {"frame-labels": FrameLabelsTarget()}
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_derived_paths(cls, data):
        if not isinstance(data, dict):
            return data
        root = data.get("root")
        if root and not data.get("shards_url"):
            # TODO: pick shards for `split` once the on-disk layout is finalized.
            data["shards_url"] = f"{root}/shards/shard_{{000000..000005}}.tar"
        if root:
            data.setdefault("video_path", f"{root}/videos.tar")
            data.setdefault("video_index_path", f"{root}/videos.tar.index.json")
        return data
