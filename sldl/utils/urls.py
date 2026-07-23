from urllib.parse import urlsplit


def normalize_shard_url(url: str) -> str:
    """Prefix a local Windows path with `file:` so that webdataset's `gopen`
    can resolve it.

    `gopen` picks a handler based on `urlsplit(url).scheme`. On Windows, a
    bare path like `F:/datasets/foo.tar` has its drive letter misread as a
    one-letter scheme (`f`), which has no registered handler. Prefixing
    `file:` makes it resolve to the `file` scheme instead, with the drive
    letter preserved in the path.
    """
    scheme = urlsplit(url).scheme
    if len(scheme) == 1:
        return f"file:{url}"
    return url


def normalize_shards_url(shards_url: str | list[str]) -> str | list[str]:
    """Apply `normalize_shard_url` to one shard URL or a list of them.

    Args:
        shards_url: A single shard URL/pattern, or a list of them.

    Returns:
        The normalized URL(s), in the same shape as the input.
    """
    if isinstance(shards_url, (list, tuple)):
        return [normalize_shard_url(url) for url in shards_url]
    return normalize_shard_url(shards_url)
