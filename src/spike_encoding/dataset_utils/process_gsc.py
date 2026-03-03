import struct
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()

    fmt = {1: "b", 2: "h", 4: "i"}[sample_width]
    samples = np.array(struct.unpack(f"{n_frames * n_channels}{fmt}", raw), dtype=np.float32)

    if n_channels > 1:
        samples = samples[::n_channels]

    max_val = float(2 ** (8 * sample_width - 1))
    samples /= max_val
    return samples


def _pad_or_truncate(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) >= length:
        return samples[:length]
    return np.pad(samples, (0, length - len(samples)))


def process_gsc(data_dir: Path) -> tuple:
    data_dir = Path(data_dir)

    class_names = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    valid_files = set((data_dir / "validation_list.txt").read_text().splitlines())
    test_files = set((data_dir / "testing_list.txt").read_text().splitlines())

    splits = {"train": ([], []), "valid": ([], []), "test": ([], [])}

    for class_name in class_names:
        label = class_to_idx[class_name]
        class_dir = data_dir / class_name
        for wav_path in sorted(class_dir.glob("*.wav")):
            rel = f"{class_name}/{wav_path.name}"
            if rel in test_files:
                split = "test"
            elif rel in valid_files:
                split = "valid"
            else:
                split = "train"
            samples = _pad_or_truncate(_load_wav(wav_path), SAMPLE_RATE)
            splits[split][0].append(samples)
            splits[split][1].append(label)

    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    X_test, y_test = splits["test"]

    metadata = {
        "sample_rate": SAMPLE_RATE,
        "n_classes": len(class_names),
        "class_labels": class_names,
        "n_samples": len(X_train) + len(X_valid) + len(X_test),
    }

    return (
        X_train, np.array(y_train, dtype=np.int64),
        X_valid, np.array(y_valid, dtype=np.int64),
        X_test, np.array(y_test, dtype=np.int64),
        metadata,
    )
