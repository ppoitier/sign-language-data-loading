def load_bytes_from_tar(tar_path: str, offset: int, size: int) -> bytes:
    """Read a byte range from a tar archive.

    Args:
        tar_path: Path to the `.tar` file.
        offset: Byte offset of the member's content within the archive
            (e.g. from a tar index).
        size: Number of bytes to read.

    Returns:
        The raw bytes read from the archive.
    """
    with open(tar_path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    return data
