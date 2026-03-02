import pandas as pd
from tqdm import tqdm
import webdataset as wds

from sldl.utils.windows import convert_samples_to_windows
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
        use_windows: bool = False,
        window_size: int = 3000,
        window_stride: int = 2800,
        max_empty_windows: int | None = None,
        show_loading_progress: bool = False,
    ):
        self.pose_transforms = pose_transform
        self.video_transform = video_transform
        self.annotation_transform = annotation_transform
        self.targets = targets

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

    def _build_windows(
        self, window_size: int, window_stride: int, max_empty_windows: int | None = None
    ):
        n_instances = len(self.samples)
        self.samples = convert_samples_to_windows(
            self.samples, window_size, window_stride
        )
        n_windows = len(self.samples)
        print(f"Converted {n_instances} samples to {n_windows} windowed samples.")
        # TODO: max empty windows
        # if max_empty_windows is None:
        #     return
        # self.samples = filter_windows_without_annotations(
        #     self.samples, max_empty_windows
        # )
        # print(
        #     f"Removed {n_windows - len(self.samples)} empty samples. There are {len(self.samples)} final samples."
        # )

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

        if self.pose_transforms is not None:
            sample["poses"] = self.pose_transforms(sample["poses"])
        if self.use_windows:
            sample["window_id"] = f"{sample['id']}_{sample['start']}_{sample['end']}"

        if self.targets:
            sample["targets"] = {}
            for target_name, encoder in self.targets.items():
                sample["targets"][target_name] = encoder.encode(sample)

        return sample


if __name__ == "__main__":
    dataset = SignLanguageDataset(
        shards_url="file:E:/datasets/sign-language/lsfb-cont/shards/shard_000000.tar",
        use_windows=True,
        window_size=1500,
        window_stride=1000,
        show_loading_progress=True,
    )
    print("#samples = ", len(dataset))
    sample = dataset[1]
    print(type(sample))
    print("sample keys = ", sample.keys())
    print("sample #frames = ", sample["n_frames"])
    print("sample pose shapes = ", {b: p.shape for b, p in sample["poses"].items()})
