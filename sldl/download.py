from pathlib import Path

from huggingface_hub import snapshot_download


def download_lsfb_isol(
    dest_dir: str | Path = ".",
    folder_name: str = "lsfb-isol",
    dry_run: bool = False,
):
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)
    snapshot_download(
        repo_id="ppoitier/lsfb-isol",
        repo_type="dataset",
        local_dir=dest_dir / folder_name,
        dry_run=dry_run,
    )
