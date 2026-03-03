from .dataset_utils.download_shd import load_shd
from .dataset_utils.process_shd import process_shd_to_sparse
from .dataset_utils.download_gsc import load_gsc
from .dataset_utils.process_gsc import process_gsc


def load_processed_shd(n_timesteps=100):
    train_path, test_path = load_shd()
    return process_shd_to_sparse(train_path, test_path, n_timesteps)


def load_processed_gsc(version=2):
    data_dir = load_gsc(version)
    return process_gsc(data_dir)


__all__ = [
    "load_shd",
    "process_shd_to_sparse",
    "load_processed_shd",
    "load_gsc",
    "process_gsc",
    "load_processed_gsc",
]

