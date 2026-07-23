from typing import Literal

from pydantic import Field, model_validator

from sldl.configs.base import SignLanguageDatasetConfig
from sldl.targets.frame_labels import FrameLabelsTarget
from sldl.targets.target import TargetEncoder

LSFBContSplit = Literal["training", "validation", "testing", "all"]

_BODY_PARTS = (
    "upper_pose",
    "left_hand",
    "right_hand",
    "lips",
    "left_eye",
    "right_eye",
)
# Other potential body parts (not by default) are right/left_iris, right/left_eyebrow


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
    shards_url: str | list[str] | None = None

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
            split = data["split"]
            if split == "training":
                shard_pattern = "shard_{000002..000007}.tar"
            elif split == "validation":
                shard_pattern = "shard_000001.tar"
            elif split == "testing":
                shard_pattern = "shard_000000.tar"
            elif split == "all":
                shard_pattern = "shard_{000000..000007}.tar"
            else:
                raise ValueError(f"Unknown split: {split}.")
            data["shards_url"] = f"{root}/shards/annotated/" + shard_pattern
        if root:
            data.setdefault("video_path", f"{root}/videos.tar")
            data.setdefault("video_index_path", f"{root}/videos.tar.index.json")
        return data

    @model_validator(mode="after")
    def _check_shards_url(self) -> "LSFBContConfig":
        if not self.shards_url:
            raise ValueError("`shards_url` could not be derived; check `root`/`variant`/`split`.")
        return self
