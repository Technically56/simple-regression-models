import hashlib, os, shutil, tarfile, urllib.request
from pathlib import Path
from base import get_base_path

PROJECT_ROOT = get_base_path()
DEFAULT_URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
DEFAULT_SHA256 = "b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812" 

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def _progress(blocks: int, block_size: int, total: int):
    size = f"{total / 1e9:.2f} GB" if total > 0 else "unknown"   # urlretrieve passes -1 when the server sends no size
    print(f"\r{blocks * block_size / 1e9:.2f} GB / {size}", end="", flush=True)


def download_ember(dest:Path | None = None, url:str | None = None, sha256:str | None = None, keep_archive:bool=False) -> Path:
    """Download, verify and extract EMBER into dest. Returns dest."""
    dest = Path(dest or os.environ.get("EMBER_DEST", "dataset/ember2018"))
    if not dest.is_absolute():
        dest = PROJECT_ROOT / dest
    if dest.exists():
        return dest

    url = url or os.environ.get("EMBER_URL", DEFAULT_URL)
    expected = (sha256 or os.environ.get("EMBER_SHA256", DEFAULT_SHA256)).lower()
    tmp = dest.parent / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / url.rsplit("/", 1)[-1]

    if not archive.exists() or _sha256(archive) != expected:
        part = archive.with_name(archive.name + ".part")   # unfinished downloads never look complete
        print("Downloading", url)
        urllib.request.urlretrieve(url, part, _progress)
        print()
        if _sha256(part) != expected:
            part.unlink()
            raise RuntimeError("Checksum mismatch. Check EMBER_URL / EMBER_SHA256.")
        part.replace(archive)

    staging = tmp / "extract"
    shutil.rmtree(staging, ignore_errors=True)               # clear leftovers from an interrupted run
    print("Extracting (bz2 is slow, this can take a while)...")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(staging, filter="data")
    (top,) = staging.iterdir()                               # the archive's single top folder, e.g. ember2018/
    top.rename(dest)

    shutil.rmtree(staging)
    if not keep_archive:
        archive.unlink()
        if not any(tmp.iterdir()):
            tmp.rmdir()
    return dest


if __name__ == "__main__":
    download_ember()