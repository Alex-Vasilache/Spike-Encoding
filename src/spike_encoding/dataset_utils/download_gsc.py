import tarfile
import urllib.request
from pathlib import Path


def load_gsc(version=2) -> Path:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    ver_str = f"v0.0{version}"
    dataset_dir = data_dir / f"speech_commands_{ver_str}"

    if dataset_dir.exists():
        print(f"GSC dataset already exists: {dataset_dir}")
        return dataset_dir

    url = f"https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_{ver_str}.tar.gz"
    tar_path = data_dir / f"speech_commands_{ver_str}.tar.gz"

    print(f"Downloading GSC {ver_str} dataset...")
    try:
        urllib.request.urlretrieve(url, tar_path)
        print(f"Downloaded: {tar_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

    dataset_dir.mkdir()
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(dataset_dir)
        print(f"Extracted to: {dataset_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract {tar_path}: {e}")

    try:
        tar_path.unlink()
        print(f"Cleaned up: {tar_path}")
    except Exception as e:
        print(f"Warning: Could not delete {tar_path}: {e}")

    print(f"GSC dataset ready in: {dataset_dir}")
    return dataset_dir
