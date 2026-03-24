from pathlib import Path
import json
import warnings

import pandas as pd
from tqdm import tqdm
import webdataset as wds

from sldl.utils.windows import convert_samples_to_windows, filter_empty_windows
from sldl.utils.videos import load_video_in_dir, load_video_in_tar
from sldl.targets.target import TargetEncoder


def _get_continuous_webdataset_mapping_fn(body_parts, annotations):
    # annotations: dict[str, list[str] | None] or None
    def mapping_fn(sample: dict) -> dict:
        sample = {
            "id": sample["__key__"],
            "poses": {
                body_part: sample[f"pose.{body_part}.npy"] for body_part in body_parts
            },
        }
        if annotations:
            sample["annotations"] = {
                annot_id: pd.DataFrame(
                    sample[f"annotations.{annot_id}.json"],
                    columns=columns,
                )
                for annot_id, columns in annotations.items()
            }
        sample["n_frames"] = next(iter(sample["poses"].values())).shape[0]
        return sample

    return mapping_fn


def _get_isolated_webdataset_mapping_fn(body_parts):
    def mapping_fn(sample: dict) -> dict:
        sample = {
            "id": sample["__key__"],
            "poses": {
                body_part: sample[f"pose.{body_part}.npy"] for body_part in body_parts
            },
            "label_id": int(sample["label.idx"]),
            "label": sample["label.txt"].strip(),
        }
        sample["n_frames"] = next(iter(sample["poses"].values())).shape[0]
        return sample

    return mapping_fn


class SignLanguageDataset:
    """A dataset for sign language pose data stored as WebDataset shards.

    Supports two modes:

    - **Continuous** (default): each sample contains multi-frame pose sequences
      with temporal annotations (e.g. gloss boundaries). Samples can optionally
      be split into overlapping windows.
    - **Isolated** (``isolated=True``): each sample is a single sign clip with
      a ``label.idx`` / ``label.txt`` pair instead of annotations. Windowing is
      not supported in this mode.

    In both modes, videos can be loaded on-the-fly from a directory or a
    ``.tar`` archive, and pose / video / annotation transforms are applied at
    ``__getitem__`` time.

    Args:
        shards_url: WebDataset shard pattern
            (e.g. ``"data/shards-{000..003}.tar"``).
        isolated: If ``True``, use isolated-sign mode with per-sample labels
            instead of temporal annotations.
        body_parts: Pose body parts to load from the shards. Each body part
            should have a corresponding ``pose.<body_part>.npy`` file in the
            shard.
        annotations: Annotation identifiers to load (continuous mode only).
            Each should have a corresponding ``annotations.<id>.json`` file.
        pose_transform: A callable applied to the pose dict at
            ``__getitem__`` time.
        video_transform: A callable applied to loaded video tensors at
            ``__getitem__`` time.
        annotation_transform: A callable applied to annotations (reserved for
            future use).
        targets: A mapping of target names to :class:`TargetEncoder` instances.
            Each encoder is called on the sample to produce a target value.
        precompute_targets: If ``True``, encode all targets once at load time
            rather than on every ``__getitem__`` call.
        load_videos: If ``True``, load video frames at ``__getitem__`` time.
            Requires ``video_path`` to be set.
        video_path: Path to a directory of video files or a ``.tar`` archive.
        video_index_path: Path to a JSON index for the video ``.tar`` archive.
            Defaults to ``<video_path>.index.json`` when ``video_path`` is a
            ``.tar`` file.
        use_windows: If ``True``, split continuous samples into overlapping
            temporal windows. Ignored (with a warning) in isolated mode.
        window_size: Window length in milliseconds.
        window_stride: Stride between consecutive windows in milliseconds.
        max_empty_windows: If set, discard windows that contain no annotations
            beyond this count.
        show_loading_progress: If ``True``, display a ``tqdm`` progress bar
            while loading shards.

    Example::

        # Continuous dataset with windowing
        ds = SignLanguageDataset(
            "data/shards-{000..003}.tar",
            use_windows=True,
            window_size=3000,
            window_stride=2800,
        )

        # Isolated dataset
        ds = SignLanguageDataset(
            "data/isolated-{000..001}.tar",
            isolated=True,
        )
        sample = ds[0]
        print(sample["label"])  # e.g. "HELLO"
    """

    def __init__(
        self,
        shards_url: str | list[str],
        isolated: bool = False,
        body_parts=("upper_pose", "left_hand", "right_hand"),
        annotations: tuple[str, ...] | dict[str, list[str] | None] | None = (
            "both_hands",
        ),
        pose_transform=None,
        video_transform=None,
        annotation_transform=None,
        targets: dict[str, TargetEncoder] | None = None,
        precompute_targets: bool = False,
        load_videos: bool = False,
        video_path: str | Path | None = None,
        video_index_path: str | Path | None = None,
        use_windows: bool = False,
        window_size: int = 3000,
        window_stride: int = 2800,
        max_empty_windows: int | None = None,
        show_loading_progress: bool = False,
    ):
        self.isolated = isolated
        self.pose_transforms = pose_transform
        self.video_transform = video_transform
        self.annotation_transform = annotation_transform
        self.load_videos = load_videos
        self.video_path = Path(video_path) if video_path else None
        self.video_index_path = Path(video_index_path) if video_index_path else None

        self.body_parts = body_parts
        if isinstance(annotations, (list, tuple)):
            annotations = {annot_id: None for annot_id in annotations}
        self.annotation_ids = annotations

        self.is_tar_video = False
        self.tar_index = {}

        if self.load_videos:
            if not self.video_path:
                raise ValueError(
                    "You need to specify a video_path (a directory or a .tar file)."
                )
            if self.video_path.suffix == ".tar":
                self.is_tar_video = True
                index_path = self.video_index_path or self.video_path.with_name(
                    f"{self.video_path.name}.index.json"
                )
                if not index_path.exists():
                    raise FileNotFoundError(f"Tar index not found at: {index_path}")
                with index_path.open("r") as f:
                    self.tar_index = json.load(f)
            elif self.video_index_path is not None:
                warnings.warn(
                    "`video_index_path` was provided, but `video_path` is not a .tar file. The index will be ignored."
                )

        if isolated:
            if use_windows:
                warnings.warn(
                    "Windowing is not supported for isolated datasets. Ignoring `use_windows`."
                )
                use_windows = False
            mapping_fn = _get_isolated_webdataset_mapping_fn(body_parts)
        else:
            mapping_fn = _get_continuous_webdataset_mapping_fn(body_parts, annotations)

        web_dataset = wds.DataPipeline(
            wds.SimpleShardList(shards_url),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(mapping_fn),
        )
        self.samples: list[dict] = []
        progress_bar = tqdm(
            web_dataset,
            disable=not show_loading_progress,
            unit="samples",
            desc="Loading samples",
        )
        for sample in progress_bar:
            self.samples.append(sample)
        print(f"Loaded {len(self.samples)} samples.")

        self.use_windows = use_windows
        if use_windows:
            self._build_windows(window_size, window_stride, max_empty_windows)

        self.targets = targets
        self.precompute_targets = precompute_targets
        if targets and precompute_targets:
            print("Precomputing targets...")
            for sample in self.samples:
                sample["targets"] = {}
                for target_name, encoder in targets.items():
                    sample["targets"][target_name] = encoder.encode(sample)
            print("Target precomputed.")

    def _build_windows(
        self, window_size: int, window_stride: int, max_empty_windows: int | None = None
    ):
        n_instances = len(self.samples)
        self.samples = convert_samples_to_windows(
            self.samples, window_size, window_stride
        )
        n_windows = len(self.samples)
        print(f"Converted {n_instances} samples to {n_windows} windowed samples.")
        if max_empty_windows is None:
            return
        if not self.annotation_ids:
            warnings.warn(
                "max_empty_windows is set, but no annotations were found. This parameter is therefore ignored."
            )
            return
        self.samples = filter_empty_windows(self.samples, max_empty_windows)
        print(
            f"Removed {n_windows - len(self.samples)} empty samples. There are {len(self.samples)} final samples."
        )

    # def get_label_occurrences(self):
    #     return _compute_label_occurrences(self.samples)
    #
    # def get_label_frequencies(self):
    #     occurrences = self.get_label_occurrences()
    #     return occurrences / occurrences.sum()
    #
    # def get_label_weights(self):
    #     return 1 / self.get_label_frequencies()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = {**self.samples[index]}

        if self.load_videos:
            if self.is_tar_video:
                sample["video"] = load_video_in_tar(
                    sample["id"],
                    self.video_path,
                    self.tar_index,
                    start_frame=sample.get("start"),
                    end_frame=sample.get("end"),
                )
            else:
                sample["video"] = load_video_in_dir(
                    sample["id"],
                    self.video_path,
                    start_frame=sample.get("start"),
                    end_frame=sample.get("end"),
                )
            if self.video_transform:
                sample["video"] = self.video_transform(sample["video"])

        if self.pose_transforms is not None:
            sample["poses"] = self.pose_transforms(sample["poses"])
        if self.use_windows:
            sample["window_id"] = f"{sample['id']}_{sample['start']}_{sample['end']}"

        if self.targets and not self.precompute_targets:
            sample["targets"] = {}
            for target_name, encoder in self.targets.items():
                sample["targets"][target_name] = encoder.encode(sample)

        return sample
