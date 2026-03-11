from pathlib import Path

try:
    import torchcodec
    _HAS_TORCHCODEC = True
except ImportError:
    _HAS_TORCHCODEC = False


def load_video_in_dir(sample_id: str, video_dir: str, start_frame: int | None = None, end_frame: int | None = None):
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
