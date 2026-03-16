from pathlib import Path

from sldl.utils.tar import load_bytes_from_tar

try:
    import torchcodec
    _HAS_TORCHCODEC = True
except ImportError:
    _HAS_TORCHCODEC = False


def load_video_in_dir(
        sample_id: str,
        video_dir: str | Path,
        start_frame: int | None = None,
        end_frame: int | None = None,
):
    if not _HAS_TORCHCODEC:
        raise ImportError(
            "Loading videos requires the 'torchcodec' package. "
            "Please install it using 'pip install torchcodec'."
        )
    video_path = Path(video_dir) / f"{sample_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Could not find video file: {video_path}")

    # 'dimension_order' defaults to "NCHW" (Batch, Channels, Height, Width).
    # You can change it to "NHWC" if your transforms prefer channels-last.
    decoder = torchcodec.decoders.VideoDecoder(str(video_path), dimension_order="NCHW")

    # Slicing the decoder returns a highly optimized torch.Tensor.
    video_tensor = decoder[start_frame:end_frame]
    return video_tensor


def load_video_in_tar(
    sample_id: str,
    tar_path: str | Path,
    tar_index: dict,
    start_frame: int | None = None,
    end_frame: int | None = None,
):
    if not _HAS_TORCHCODEC:
        raise ImportError(
            "Loading videos requires the 'torchcodec' package. "
            "Please install it using 'pip install torchcodec'."
        )

    # Check both with and without the .mp4 extension
    video_key = f"{sample_id}.mp4" if f"{sample_id}.mp4" in tar_index else sample_id
    if video_key not in tar_index:
        raise KeyError(f"Video ID '{video_key}' not found in the tar index.")

    video_offset, video_size = tar_index[video_key]
    video_bytes = load_bytes_from_tar(str(tar_path), video_offset, video_size)
    # torchcodec can decode directly from an in-memory bytes object
    decoder = torchcodec.decoders.VideoDecoder(video_bytes, dimension_order="NCHW")
    video_tensor = decoder[start_frame:end_frame]
    return video_tensor
