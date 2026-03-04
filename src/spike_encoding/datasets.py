import json

import numpy as np

from .dataset_utils.download_shd import load_shd
from .dataset_utils.process_shd import process_shd_to_sparse
from .dataset_utils.download_gsc import load_gsc
from .dataset_utils.process_gsc import process_gsc


def load_processed_shd(n_timesteps=100):
    train_path, test_path = load_shd()
    return process_shd_to_sparse(train_path, test_path, n_timesteps)


_GSC_ARRAY_NAMES = ["X_train", "y_train", "X_valid", "y_valid", "X_test", "y_test"]


def load_processed_gsc(version=2):
    data_dir = load_gsc(version)
    cache_dir = data_dir.parent / f"{data_dir.name}_processed"

    if cache_dir.exists() and all((cache_dir / f"{f}.npy").exists() for f in _GSC_ARRAY_NAMES):
        print(f"Loading cached GSC from {cache_dir}")
        arrays = [np.load(cache_dir / f"{f}.npy") for f in _GSC_ARRAY_NAMES]
        metadata = json.loads((cache_dir / "metadata.json").read_text())
        return (*arrays, metadata)

    result = process_gsc(data_dir)
    *arrays, metadata = result
    cache_dir.mkdir(exist_ok=True)
    for name, arr in zip(_GSC_ARRAY_NAMES, arrays):
        np.save(cache_dir / f"{name}.npy", np.array(arr))
    (cache_dir / "metadata.json").write_text(json.dumps(metadata))
    print(f"Cached processed GSC to {cache_dir}")
    return result


__all__ = [
    "load_shd",
    "process_shd_to_sparse",
    "load_processed_shd",
    "load_gsc",
    "process_gsc",
    "load_processed_gsc",
]

