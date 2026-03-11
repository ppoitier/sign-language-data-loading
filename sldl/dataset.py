import pandas as pd
from tqdm import tqdm
import webdataset as wds

from sldl.utils.windows import convert_samples_to_windows, filter_empty_windows
from sldl.utils.videos import load_video_in_dir
from sldl.targets.target import TargetEncoder


def _get_webdataset_mapping_fn(body_parts, annotations):
    def mapping_fn(sample: dict) -> dict:
        sample = {
            "id": sample["__key__"],
            "poses": {
                body_part: sample[f"pose.{body_part}.npy"] for body_part in body_parts
            },
            "annotations": {
                annot_id: pd.DataFrame(
                    sample[f"annotations.{annot_id}.json"],
                    columns=[
                        "start_ms",
                        "end_ms",
                        "gloss",
                        "start_frame",
                        "end_frame",
                        "lemma",
                        "sign_type",
                        "specifier",
                        "variation",
                    ],
                )
                for annot_id in annotations
            },
        }
        sample["n_frames"] = next(iter(sample["poses"].values())).shape[0]
        return sample

    return mapping_fn


class SignLanguageDataset:

    def __init__(
        self,
        shards_url: str,
        body_parts=("upper_pose", "left_hand", "right_hand"),
        annotations=("both_hands",),
        pose_transform=None,
        video_transform=None,
        annotation_transform=None,
        targets: dict[str, TargetEncoder] | None = None,
        precompute_targets: bool = False,
        load_videos: bool = False,
        video_dir: str | None = None,
        use_windows: bool = False,
        window_size: int = 3000,
        window_stride: int = 2800,
        max_empty_windows: int | None = None,
        show_loading_progress: bool = False,
    ):
        self.pose_transforms = pose_transform
        self.video_transform = video_transform
        self.annotation_transform = annotation_transform
        self.load_videos = load_videos
        self.video_dir = video_dir
        if self.load_videos and not self.video_dir:
            raise ValueError("You need to specify a video directory.")

        web_dataset = wds.DataPipeline(
            wds.SimpleShardList(shards_url),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(_get_webdataset_mapping_fn(body_parts, annotations)),
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
            sample["video"] = load_video_in_dir(
                sample["id"],
                self.video_dir,
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
