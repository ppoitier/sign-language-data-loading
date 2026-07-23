from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sldl.targets.target import TargetEncoder


class SignLanguageDatasetConfig(BaseModel):
    """Configuration for :class:`sldl.dataset.SignLanguageDataset`.

    Mirrors the constructor parameters of ``SignLanguageDataset``. Instantiate
    directly, or subclass it to define reusable presets for specific datasets
    (overriding only the fields that differ from the defaults)::

        class MyDatasetContinuousConfig(SignLanguageDatasetConfig):
            shards_url: str = "data/my-dataset-{000..003}.tar"
            annotations: dict[str, list[str] | None] = {"both_hands": ["lemma"]}

        ds = SignLanguageDataset.from_config(MyDatasetContinuousConfig())
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    shards_url: str | list[str] = Field(
        description="WebDataset shard pattern (e.g. 'data/shards-{000..003}.tar')."
    )
    isolated: bool = Field(
        default=False,
        description="If True, use isolated-sign mode with per-sample labels instead of temporal annotations.",
    )
    body_parts: tuple[str, ...] = Field(
        default=("upper_pose", "left_hand", "right_hand"),
        description="Pose body parts to load from the shards.",
    )
    annotations: tuple[str, ...] | dict[str, list[str] | None] | None = Field(
        default=("both_hands",),
        description="Annotation identifiers to load (continuous mode only).",
    )
    pose_transform: Callable | None = Field(
        default=None, description="A callable applied to the pose dict at __getitem__ time."
    )
    video_transform: Callable | None = Field(
        default=None, description="A callable applied to loaded video tensors at __getitem__ time."
    )
    annotation_transform: Callable | None = Field(
        default=None, description="A callable applied to annotations (reserved for future use)."
    )
    targets: dict[str, TargetEncoder] | None = Field(
        default=None, description="A mapping of target names to TargetEncoder instances."
    )
    precompute_targets: bool = Field(
        default=False,
        description="If True, encode all targets once at load time rather than on every __getitem__ call.",
    )
    load_videos: bool = Field(
        default=False, description="If True, load video frames at __getitem__ time."
    )
    video_path: str | Path | None = Field(
        default=None, description="Path to a directory of video files or a .tar archive."
    )
    video_index_path: str | Path | None = Field(
        default=None, description="Path to a JSON index for the video .tar archive."
    )
    use_windows: bool = Field(
        default=False,
        description="If True, split continuous samples into overlapping temporal windows.",
    )
    window_size: int = Field(default=3000, description="Window length in milliseconds.")
    window_stride: int = Field(
        default=2800, description="Stride between consecutive windows in milliseconds."
    )
    max_empty_windows: int | None = Field(
        default=None,
        description="If set, discard windows that contain no annotations beyond this count.",
    )
    show_loading_progress: bool = Field(
        default=False, description="If True, display a tqdm progress bar while loading shards."
    )

    @model_validator(mode="after")
    def _check_video_path(self) -> "SignLanguageDatasetConfig":
        if self.load_videos and not self.video_path:
            raise ValueError("`video_path` must be set when `load_videos=True`.")
        return self
