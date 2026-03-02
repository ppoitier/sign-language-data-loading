from typing import Any

from sldl.targets.target import TargetEncoder

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class SignLanguageCollator:
    def __init__(
        self,
        create_masks: bool = True,
        pad_value: float = 0.0,
        targets: dict[str, TargetEncoder] | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Please install it using 'pip install torch' "
                "to use the SignLanguageCollator."
            )
        self.create_masks: bool = create_masks
        self.pad_value: float = pad_value
        self.targets = targets or {}

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        keys = batch[0].keys()
        collated: dict[str, Any] = {k: [b[k] for b in batch] for k in keys}

        # Cast standard integers to tensors
        if "n_frames" in collated:
            collated["n_frames"] = torch.tensor(collated["n_frames"], dtype=torch.long)
        if "start" in collated:
            collated["start"] = torch.tensor(collated["start"], dtype=torch.long)
        if "end" in collated:
            collated["end"] = torch.tensor(collated["end"], dtype=torch.long)

        # Process poses if they exist
        if "poses" in collated and collated["poses"][0] is not None:
            poses_list: list[Any] = collated["poses"]
            sample_pose = poses_list[0]

            # 1. Calculate lengths upfront
            if isinstance(sample_pose, dict):
                first_bp = next(iter(sample_pose.keys()))
                lengths = torch.tensor(
                    [len(p[first_bp]) for p in poses_list], dtype=torch.long
                )
            else:
                lengths = torch.tensor([len(p) for p in poses_list], dtype=torch.long)

            # 2. Pad sequences
            if isinstance(sample_pose, dict):
                collated["poses"] = {}
                for bp in sample_pose.keys():
                    tensors = [torch.as_tensor(p[bp]) for p in poses_list]
                    collated["poses"][bp] = pad_sequence(
                        tensors, batch_first=True, padding_value=self.pad_value
                    )
            else:
                tensors = [torch.as_tensor(p) for p in poses_list]
                collated["poses"] = pad_sequence(
                    tensors, batch_first=True, padding_value=self.pad_value
                )

            # 3. Create masks using the pre-calculated lengths
            if self.create_masks:
                max_len = lengths.max().item()
                collated["masks"] = torch.arange(max_len) < lengths.unsqueeze(1)
                collated["lengths"] = lengths

        if self.targets and "targets" in collated:
            batch_targets = collated["targets"]
            final_targets = {}
            for target_name, encoder in self.targets.items():
                final_targets[target_name] = encoder.collate([b[target_name] for b in batch_targets])
            collated["targets"] = final_targets

        return collated
