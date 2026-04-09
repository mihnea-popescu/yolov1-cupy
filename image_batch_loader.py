import random
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image

_IMAGE_EXT = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG"}
)


def load_image_batch(image_paths, size=(224, 224)):
    """Load paths as one CuPy batch: (N, 3, H, W), values in [0, 1], channel-first."""
    chw_list = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        hwc = np.asarray(img, dtype=np.float64) / 255.0
        chw = np.transpose(hwc, (2, 0, 1))
        chw_list.append(chw)
    batch_np = np.stack(chw_list, axis=0)
    return cp.asarray(batch_np)


def _labeled_paths_from_repo(
    repo_root,
    data_root=None,
    try_relative=("imagenet10/train", "train", "imagenet10", "val"),
):
    repo_root = Path(repo_root)
    if data_root is not None:
        root = Path(data_root)
    else:
        root = repo_root
        for rel in try_relative:
            candidate = repo_root / rel
            if candidate.is_dir():
                root = candidate
                break

    paths = sorted(
        p
        for p in root.rglob("*")
        if p.suffix in _IMAGE_EXT and p.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No images under {root}")

    class_names = sorted({p.parent.name for p in paths})
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    labeled = [(p, class_to_idx[p.parent.name]) for p in paths]
    return labeled


def _tensor_pair_from_labeled(labeled_chunk, size):
    paths_only = [p for p, _ in labeled_chunk]
    y = np.array([y for _, y in labeled_chunk], dtype=np.int64)
    x = load_image_batch([str(p) for p in paths_only], size=size)
    return x, y


def image_label_batch(
    repo_root,
    batch_size=8,
    *,
    seed=0,
    batch_index=0,
    data_root=None,
    size=(224, 224),
):
    """
    Find images under ``repo_root`` (or ``data_root``), shuffle with ``seed``,
    return one CuPy batch ``(N,3,H,W)`` and NumPy labels ``(N,)``.

    Increase ``batch_index`` for the next slice in the same shuffled order;
    change ``seed`` each epoch to reshuffle.
    """
    labeled = _labeled_paths_from_repo(repo_root, data_root=data_root)
    rng = random.Random(seed)
    rng.shuffle(labeled)
    start = batch_index * batch_size
    chunk = labeled[start : start + batch_size]
    if not chunk:
        raise ValueError(
            f"No samples for batch_index={batch_index} (dataset size {len(labeled)})"
        )
    return _tensor_pair_from_labeled(chunk, size)


def dataset_size(repo_root, data_root=None) -> int:
    """Number of labeled images found under the repo (or ``data_root``)."""
    return len(_labeled_paths_from_repo(repo_root, data_root=data_root))


def num_batches_per_epoch(repo_root, batch_size, data_root=None) -> int:
    """Batches when covering every sample once (last batch may be smaller)."""
    n = dataset_size(repo_root, data_root=data_root)
    if n == 0:
        return 0
    return (n + batch_size - 1) // batch_size
