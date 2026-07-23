<p align="center">
  <img src="docs/assets/banner.png" alt="sign-language-data-loading" width="720">
</p>

# sign-language-data-loading (sldl)

Efficient, PyTorch-friendly data loading for sign language datasets stored as [WebDataset](https://github.com/webdataset/webdataset) shards — poses, videos, and temporal annotations, with windowing and target encoding built in.

## Installation

You can simply install it in your Python environment:
```bash
pip install sign-language-data-loading
```
The package is imported as `sldl` (see examples).

Some features need extra dependencies. Typically, data-loaders and custom targets need PyTorch.
If you load the videos too, torchcodec is required.
You can install them along with:
```bash
# For SignLanguageCollator and the built-in target encoders (sldl.targets.*)
pip install "sign-language-data-loading[torch]"

# For on-the-fly video loading (load_videos=True)
pip install "sign-language-data-loading[video]"
```

However, we do recommend setting them up yourself in your environment before installing sign-language-data-loading.



## Quickstart

### Load an isolated-sign dataset

```python
from sldl import SignLanguageDataset
from sldl.configs import LSFBIsolConfig

dataset = SignLanguageDataset.from_config(
    LSFBIsolConfig(
        root="path/to/lsfb-isol",
        variant="500",
        split="training",
    )
)

sample = dataset[0]
print(sample["label"])          # e.g. "bonjour"
print(sample["poses"]["upper_pose"].shape)
```

### Load a continuous dataset

```python
from sldl import SignLanguageDataset
from sldl.configs import LSFBContConfig

dataset = SignLanguageDataset.from_config(
    LSFBContConfig(
        root="path/to/lsfb-cont",
        split="training",
    )
)

sample = dataset[0]
print(sample["annotations"]["both_hands"].head())
```

### Batch samples for training

```python
from torch.utils.data import DataLoader
from sldl import SignLanguageCollator

loader = DataLoader(dataset, batch_size=8, collate_fn=SignLanguageCollator())
batch = next(iter(loader))
print(batch["poses"]["upper_pose"].shape)   # (batch, max_n_frames, ...)
print(batch["masks"].shape)                 # (batch, max_n_frames)
```

## Going further

- Custom dataset paths, windowing, and target encoders: see `SignLanguageDataset` and `sldl.targets`.
- Full worked examples (loading, visualizing poses/video, windowing, target encoding) are in [`examples/`](examples/).
