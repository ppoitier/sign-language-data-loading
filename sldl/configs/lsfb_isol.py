from typing import Literal

from pydantic import model_validator

from sldl.config import SignLanguageDatasetConfig

LSFBIsolVariant = Literal["500", "750", "2000", "all"]
LSFBIsolSplit = Literal["train", "eval", "test", "all"]

_BODY_PARTS = (
    "upper_pose",
    "left_hand",
    "right_hand",
    "lips",
    "left_eye",
    "right_eye",
)


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
        split = data.get("split")
        if root and variant and not data.get("shards_url"):
            if split == "train":
                data["shards_url"] = f"{root}/shards/{variant}/shard_{{000000..000006}}.tar"
            elif split == "eval":
                data["shards_url"] = f"{root}/shards/{variant}/shard_{{000007..000008}}.tar"
            elif split == "test":
                data["shards_url"] = f"{root}/shards/{variant}/shard_000009.tar"
            elif split == "all":
                data["shards_url"] = f"{root}/shards/{variant}/shard_{{000000..000009}}.tar"
            else:
                raise ValueError(f"Unknown split {split}")
        if root:
            data.setdefault("video_path", f"{root}/videos.tar")
            data.setdefault("video_index_path", f"{root}/videos.tar.index.json")
        return data

    @model_validator(mode="after")
    def _check_shards_url(self) -> "LSFBIsolConfig":
        if not self.shards_url:
            raise ValueError("`shards_url` could not be derived; check `root`/`variant`/`split`.")
        return self
