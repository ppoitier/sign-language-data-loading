from typing import Literal

from pydantic import model_validator

from sldl.configs.base import SignLanguageDatasetConfig

LSFBIsolVariant = Literal["500", "750", "2000", "all"]
LSFBIsolSplit = Literal["training", "validation", "testing", "all"]

_BODY_PARTS = (
    "upper_pose",
    "left_hand",
    "right_hand",
    "lips",
    "left_eye",
    "right_eye",
)
# Other potential body parts (not by default) are right/left_iris, right/left_eyebrow


class LSFBIsolConfig(SignLanguageDatasetConfig):
    """Config preset for the LSFB-ISOL isolated sign dataset.

    `shards_url`, `video_path` and `video_index_path` are derived from
    `root` / `variant` / `split` unless explicitly overridden.

    Example::

        config = LSFBIsolConfig(
            root="F:/datasets/sign-language/lsfb-isol",
            variant="500",
            split="train",
        )
        ds = SignLanguageDataset.from_config(config)
    """

    root: str
    variant: LSFBIsolVariant
    split: LSFBIsolSplit
    shards_url: str | list[str] | None = None

    isolated: bool = True
    body_parts: tuple[str, ...] = _BODY_PARTS
    annotations: None = None
    show_loading_progress: bool = True
    load_videos: bool = True

    @model_validator(mode="before")
    @classmethod
    def _fill_derived_paths(cls, data):
        if not isinstance(data, dict):
            return data
        root = data.get("root")
        variant = data.get("variant")
        if root and variant and not data.get("shards_url"):
            split = data["split"]
            if split == "training":
                shard_pattern = "shard_{000003..000009}.tar"
            elif split == "validation":
                shard_pattern = "shard_{000001..000002}.tar"
            elif split == "testing":
                shard_pattern = "shard_000000.tar"
            elif split == "all":
                shard_pattern = "shard_{000000..000009}.tar"
            else:
                raise ValueError(f"Unknown split: {split}.")
            data["shards_url"] = f"{root}/shards/{variant}/" + shard_pattern
        if root:
            data.setdefault("video_path", f"{root}/videos.tar")
            data.setdefault("video_index_path", f"{root}/videos.tar.index.json")
        return data

    @model_validator(mode="after")
    def _check_shards_url(self) -> "LSFBIsolConfig":
        if not self.shards_url:
            raise ValueError("`shards_url` could not be derived; check `root`/`variant`/`split`.")
        return self
