
def load_bytes_from_tar(tar_path: str, offset: int, size: int) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    return data
