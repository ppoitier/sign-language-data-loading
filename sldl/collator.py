from typing import Any

from sldl.targets.target import TargetEncoder

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


__all__ = ["SignLanguageCollator"]


def _collate_sequence(
    seqs: list[Any], pad_value: float, fixed_length: int | None = None
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Pad and stack a list of time-major sequences along a new batch dimension.

    Args:
        seqs: Per-sample sequences of shape `(T_i, ...)`. Anything accepted by
            `torch.as_tensor` (numpy arrays included) works; tensors are used
            as-is, without a copy.
        pad_value: Value written in the padded region.
        fixed_length: If given, pad every sequence to this length instead of
            the batch's own longest sequence. Raises if a sequence is longer
            than `fixed_length`. Useful to bound `T_max` to a constant across
            batches, e.g. so a `torch.compile`-d model sees a fixed number of
            distinct shapes instead of recompiling for every batch.

    Returns:
        A tuple `(padded, lengths)` where `padded` has shape
        `(B, T_max, ...)` (`T_max` being `fixed_length` when given) and
        `lengths` is a `torch.long` tensor of shape `(B,)` holding the
        original `T_i`.
    """
    if not torch.is_tensor(seqs[0]):
        seqs = [torch.as_tensor(s) for s in seqs]

    lengths = [s.shape[0] for s in seqs]
    natural_max_len = max(lengths)
    if fixed_length is not None:
        if natural_max_len > fixed_length:
            raise ValueError(
                f"Got a sequence of length {natural_max_len}, which exceeds "
                f"fixed_length={fixed_length}."
            )
        target_len = fixed_length
    else:
        target_len = natural_max_len

    # Batches where every sequence already has the target length (windowed /
    # cropped datasets, or fixed_length matching the batch exactly) skip
    # padding entirely: a single fused `stack` is cheaper than padding.
    if min(lengths) == target_len:
        padded = torch.stack(seqs, 0)
    elif target_len == natural_max_len:
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    else:
        # `pad_sequence` only pads to the longest sequence given to it, so a
        # `target_len` beyond the batch's own max needs manual placement.
        padded = seqs[0].new_full(
            (len(seqs), target_len, *seqs[0].shape[1:]), pad_value
        )
        for i, s in enumerate(seqs):
            padded[i, : s.shape[0]] = s
    return padded, torch.tensor(lengths, dtype=torch.long)


def _collate_field(
    values: list[Any], pad_value: float, fixed_length: int | None = None
) -> tuple[Any, Any]:
    """Collate one sample field, preserving its container structure.

    Dispatches on the structure of the first sample:

    - `dict` (e.g. one entry per body part): recurses per key.
    - `tuple` / `list` (e.g. multiple augmented views): recurses per view and
      returns a `tuple`, since views may have different lengths.
    - anything else: treated as a single sequence and padded.

    Args:
        values: The per-sample values of a single field, one entry per sample.
        pad_value: Value written in the padded region.
        fixed_length: If given, pad every sequence (and every view) to this
            length instead of the batch's own longest sequence.

    Returns:
        A tuple `(collated, lengths)`. `collated` mirrors the input structure
        with tensors replaced by their batched counterparts. `lengths` is a
        `(B,)` tensor, or a tuple of `(B,)` tensors when the field is
        multi-view. For dicts, the lengths of the first key are returned
        (all keys of a sample share the same temporal length).
    """
    first = values[0]

    if isinstance(first, dict):
        collated: dict[str, Any] = {}
        lengths: Any = None
        for key in first:
            collated[key], key_lengths = _collate_field(
                [v[key] for v in values], pad_value, fixed_length
            )
            if lengths is None:
                lengths = key_lengths
        return collated, lengths

    if isinstance(first, (tuple, list)):
        views, view_lengths = [], []
        for view_idx in range(len(first)):
            padded, lengths = _collate_sequence(
                [v[view_idx] for v in values], pad_value, fixed_length
            )
            views.append(padded)
            view_lengths.append(lengths)
        return tuple(views), tuple(view_lengths)

    return _collate_sequence(values, pad_value, fixed_length)


def _masks_from_lengths(lengths: Any, fixed_length: int | None = None) -> Any:
    """Build boolean padding masks from sequence lengths.

    Args:
        lengths: A `(B,)` tensor of lengths, or a tuple of such tensors (one
            per view).
        fixed_length: The `T_max` the matching field was padded to. Must
            match what was passed to `_collate_field`, otherwise the mask
            width would not agree with the padded tensor's shape. Defaults to
            `lengths.max()`, i.e. the batch's own longest sequence.

    Returns:
        A `(B, T_max)` boolean tensor that is `True` on real frames and
        `False` on padding, or a tuple of such tensors mirroring the input.
    """
    if isinstance(lengths, tuple):
        return tuple(_masks_from_lengths(length, fixed_length) for length in lengths)
    t_max = fixed_length if fixed_length is not None else lengths.max()
    return torch.arange(t_max) < lengths.unsqueeze(1)


class SignLanguageCollator:
    """Collate function that pads variable-length `SignLanguageDataset` samples into batches.

    Designed to be used as the `collate_fn` of a `torch.utils.data.DataLoader`.
    It pads every temporal field to the longest sequence in the batch, casts
    the scalar metadata to tensors, and delegates target collation to the
    `TargetEncoder` instances it was given. Requires PyTorch.

    Fields that are not recognised (`id`, `label`, `signer_id`, ...) are left
    untouched as plain Python lists.

    Shapes:
        With `B` the batch size and `T_max` the longest sequence in the batch
        (or `fixed_length`, when set):

        | Field       | Per sample     | Collated             |
        |-------------|----------------|----------------------|
        | `poses`     | `(T, L, C)`    | `(B, T_max, L, C)`   |
        | `video`     | `(T, 3, H, W)` | `(B, T_max, 3, H, W)`|
        | `masks`     | —              | `(B, T_max)`, `bool` |
        | `lengths`   | —              | `(B,)`, `long`       |

        where `T` is the number of frames, `L` the number of landmarks and `C`
        the number of coordinates per landmark.

        `poses` (and `video`) keep the container structure of the samples:

        - A plain tensor stays a tensor.
        - A `dict` of body parts (`{"upper_pose": ..., "left_hand": ...}`)
          stays a dict, each entry padded to the same `T_max`.
        - A `tuple` of augmented views (as produced by contrastive pipelines
          such as SimCLR) stays a tuple. Each view is padded independently, so
          views may have different lengths, and `masks` / `lengths` become
          tuples with one entry per view. Both `poses = (v1, v2)` and
          `poses = {"upper_pose": (v1, v2), ...}` are supported.

    Args:
        create_masks: If `True`, add a boolean `masks` tensor and a `lengths`
            tensor to the collated batch, derived from the per-sample pose
            lengths (or from the video lengths when the samples have no poses).
        pad_value: Padding value used for the `poses` tensors.
        pad_videos: If `True`, pad and stack the `video` tensors into a single
            batched tensor. If `False`, `video` is left as a plain list of
            per-sample tensors.
        video_pad_value: Padding value used for the `video` tensors. Kept
            separate from `pad_value` because poses are often padded with a
            sentinel value that would be meaningless for pixels.
        fixed_length: If given, pad every sequence (`poses` and `video`, every
            view) to this constant `T_max` instead of the batch's own longest
            sequence. Raises if a sample is longer than `fixed_length`. Useful
            to keep a `torch.compile`-d model from recompiling on every new
            sequence length it encounters.
        targets: A mapping of target names to `TargetEncoder` instances, used
            to collate the `targets` dict of each sample (via each encoder's
            `collate` method). Must match the `targets` passed to the
            `SignLanguageDataset`.

    Raises:
        ImportError: If PyTorch is not installed.

    Example:
        ```python
        from torch.utils.data import DataLoader
        from sldl import SignLanguageDataset, SignLanguageCollator

        dataset = SignLanguageDataset("data/shards-{000..003}.tar")
        loader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=SignLanguageCollator(),
        )

        batch = next(iter(loader))
        batch["poses"]["upper_pose"].shape  # (8, T_max, L, C)
        batch["masks"].shape                # (8, T_max)
        batch["lengths"]                    # tensor([...]) of true lengths

        # Contrastive setup: the pose transform returns a tuple of two views.
        loader = DataLoader(
            simclr_dataset,
            batch_size=8,
            collate_fn=SignLanguageCollator(),
        )
        view_1, view_2 = next(iter(loader))["poses"]["upper_pose"]
        ```
    """

    def __init__(
        self,
        create_masks: bool = True,
        pad_value: float = 0.0,
        pad_videos: bool = True,
        video_pad_value: float = 0.0,
        fixed_length: int | None = None,
        targets: dict[str, TargetEncoder] | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Please install it using 'pip install torch' "
                "to use the SignLanguageCollator."
            )
        self.create_masks: bool = create_masks
        self.pad_value: float = pad_value
        self.pad_videos: bool = pad_videos
        self.video_pad_value: float = video_pad_value
        self.fixed_length: int | None = fixed_length
        self.targets = targets or {}

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a list of samples into a single padded batch.

        Args:
            batch: A list of samples, as returned by
                `SignLanguageDataset.__getitem__`.

        Returns:
            A dict with the same keys as the input samples, with `poses`
            padded and stacked, `video` padded if `pad_videos=True`,
            `n_frames`/`start`/`end` cast to tensors, `targets` collated
            per-target (if `targets` was set), and, if `create_masks=True`,
            additional `masks` and `lengths` keys. See the class docstring
            for the resulting shapes.
        """
        keys = batch[0].keys()
        collated: dict[str, Any] = {k: [b[k] for b in batch] for k in keys}

        # Cast standard integers to tensors
        if "n_frames" in collated:
            collated["n_frames"] = torch.tensor(collated["n_frames"], dtype=torch.long)
        if "start" in collated:
            collated["start"] = torch.tensor(collated["start"], dtype=torch.long)
        if "end" in collated:
            collated["end"] = torch.tensor(collated["end"], dtype=torch.long)

        lengths: Any = None

        if "poses" in collated and collated["poses"][0] is not None:
            collated["poses"], lengths = _collate_field(
                collated["poses"], self.pad_value, self.fixed_length
            )

        if (
            self.pad_videos
            and "video" in collated
            and collated["video"][0] is not None
        ):
            collated["video"], video_lengths = _collate_field(
                collated["video"], self.video_pad_value, self.fixed_length
            )
            # Poses are the reference for masks; videos only fill in when the
            # batch carries no pose data.
            if lengths is None:
                lengths = video_lengths

        if self.create_masks and lengths is not None:
            collated["masks"] = _masks_from_lengths(lengths, self.fixed_length)
            collated["lengths"] = lengths

        if self.targets and "targets" in collated:
            batch_targets = collated["targets"]
            final_targets = {}
            for target_name, encoder in self.targets.items():
                final_targets[target_name] = encoder.collate(
                    [b[target_name] for b in batch_targets]
                )
            collated["targets"] = final_targets

        return collated
