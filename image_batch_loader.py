import random
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

import cupy as cp
import numpy as np
from PIL import Image

_IMAGE_EXT = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG"}
)

VOC_CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
VOC_CLASS_TO_INDEX = {name: idx for idx, name in enumerate(VOC_CLASS_NAMES)}


def load_image_batch(image_paths, size=(224, 224)):
    """Load paths as one CuPy batch: (N, 3, H, W), values in [0, 1], channel-first."""
    chw_list = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        hwc = cp.asarray(img, dtype=cp.float32) / 255.0
        chw = cp.transpose(hwc, (2, 0, 1))
        chw_list.append(chw)
    return cp.stack(chw_list, axis=0)


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
    y = cp.asarray([y for _, y in labeled_chunk], dtype=cp.int64)
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
    return one CuPy batch ``(N,3,H,W)`` and CuPy labels ``(N,)``.

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


def _find_voc_root(repo_root, data_root=None):
    repo_root = Path(repo_root)
    root = None
    if data_root is not None:
        data_root_path = Path(data_root)
        data_candidates = (
            data_root_path,
            data_root_path / "VOC2012",
            data_root_path / "VOCdevkit" / "VOC2012",
            data_root_path / "VOC2012_train_val",
            data_root_path / "VOC2012_train_val" / "VOC2012_train_val",
            data_root_path / "VOC2012_train_val" / "VOC2012",
            data_root_path / "VOC2012_train_val" / "VOCdevkit" / "VOC2012",
        )
        for candidate in data_candidates:
            if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
                root = candidate
                break
        if root is None:
            for candidate in data_root_path.rglob("VOC2012"):
                if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
                    root = candidate
                    break
    else:
        candidates = (
            repo_root / "pascal_voc_2012",
            repo_root / "VOCdevkit" / "VOC2012",
            repo_root / "VOC2012_train_val",
            repo_root / "VOC2012_train_val" / "VOC2012_train_val",
            repo_root / "VOC2012_train_val" / "VOC2012",
            repo_root / "pascal_voc_2012" / "VOCdevkit" / "VOC2012",
            repo_root / "VOC2012",
        )
        for candidate in candidates:
            if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
                root = candidate
                break
        if root is None:
            for candidate in repo_root.rglob("VOC2012"):
                if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
                    root = candidate
                    break
    if root is None:
        raise FileNotFoundError("Could not locate VOC2012 root with JPEGImages/Annotations")
    return root


def _voc_split_ids(voc_root, split="train"):
    split_file = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing VOC split file: {split_file}")
    with split_file.open("r", encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    if not ids:
        raise ValueError(f"Split file is empty: {split_file}")
    return ids


def _voc_image_annotation_pairs(repo_root, data_root=None, split="train"):
    voc_root = _find_voc_root(repo_root, data_root=data_root)
    image_ids = _voc_split_ids(voc_root, split=split)
    pairs = []
    for image_id in image_ids:
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        annotation_path = voc_root / "Annotations" / f"{image_id}.xml"
        if image_path.is_file() and annotation_path.is_file():
            pairs.append((image_path, annotation_path))
    if not pairs:
        raise FileNotFoundError(
            f"No valid image/annotation pairs found in split={split} under {voc_root}"
        )
    return pairs


def _parse_voc_annotation(annotation_path, class_to_index=None, skip_difficult=True):
    if class_to_index is None:
        class_to_index = VOC_CLASS_TO_INDEX
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    size_node = root.find("size")
    if size_node is None:
        return []
    width_node = size_node.find("width")
    height_node = size_node.find("height")
    if width_node is None or height_node is None:
        return []

    image_width = float(width_node.text)
    image_height = float(height_node.text)
    if image_width <= 1.0 or image_height <= 1.0:
        return []

    parsed_objects = []
    for object_node in root.findall("object"):
        class_name_node = object_node.find("name")
        if class_name_node is None:
            continue
        class_name = class_name_node.text.strip()
        if class_name not in class_to_index:
            continue
        if skip_difficult:
            difficult_node = object_node.find("difficult")
            difficult_value = (
                difficult_node.text.strip() if difficult_node is not None and difficult_node.text else "0"
            )
            if difficult_value == "1":
                continue

        box_node = object_node.find("bndbox")
        if box_node is None:
            continue
        xmin_node = box_node.find("xmin")
        ymin_node = box_node.find("ymin")
        xmax_node = box_node.find("xmax")
        ymax_node = box_node.find("ymax")
        if None in (xmin_node, ymin_node, xmax_node, ymax_node):
            continue

        xmin = max(0.0, float(xmin_node.text) - 1.0)
        ymin = max(0.0, float(ymin_node.text) - 1.0)
        xmax = min(image_width - 1.0, float(xmax_node.text) - 1.0)
        ymax = min(image_height - 1.0, float(ymax_node.text) - 1.0)
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) * 0.5) / image_width
        y_center = ((ymin + ymax) * 0.5) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        x_center = min(max(x_center, 0.0), 1.0 - 1e-6)
        y_center = min(max(y_center, 0.0), 1.0 - 1e-6)
        box_width = min(max(box_width, 1e-6), 1.0)
        box_height = min(max(box_height, 1e-6), 1.0)

        parsed_objects.append(
            {
                "class_index": class_to_index[class_name],
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height,
            }
        )
    return parsed_objects


def encode_yolov1_target(objects, s=7, b=2, c=20):
    """
    Encode one image's objects into YOLOv1 target tensor shape (S, S, B*5 + C).
    Layout is [x, y, w, h, conf] repeated B times, followed by class one-hot.
    """
    target = cp.zeros((s, s, b * 5 + c), dtype=cp.float32)
    box_data_end = b * 5

    for obj in objects:
        x_center = float(obj["x_center"])
        y_center = float(obj["y_center"])
        box_width = float(obj["width"])
        box_height = float(obj["height"])
        class_index = int(obj["class_index"])

        grid_x = min(int(x_center * s), s - 1)
        grid_y = min(int(y_center * s), s - 1)
        x_in_cell = x_center * s - grid_x
        y_in_cell = y_center * s - grid_y

        predictor_slot = None
        for predictor_index in range(b):
            confidence_channel = predictor_index * 5 + 4
            if float(target[grid_y, grid_x, confidence_channel].item()) == 0.0:
                predictor_slot = predictor_index
                break
        if predictor_slot is None:
            continue

        base_channel = predictor_slot * 5
        target[grid_y, grid_x, base_channel + 0] = x_in_cell
        target[grid_y, grid_x, base_channel + 1] = y_in_cell
        target[grid_y, grid_x, base_channel + 2] = box_width
        target[grid_y, grid_x, base_channel + 3] = box_height
        target[grid_y, grid_x, base_channel + 4] = 1.0
        if 0 <= class_index < c:
            target[grid_y, grid_x, box_data_end + class_index] = 1.0

    return target


def _voc_tensor_pair_from_pairs(
    pair_chunk,
    size=(448, 448),
    s=7,
    b=2,
    c=20,
    skip_difficult=True,
):
    image_paths = [str(image_path) for image_path, _ in pair_chunk]
    x = load_image_batch(image_paths, size=size).astype(cp.float32, copy=False)
    target_list = []
    for _, annotation_path in pair_chunk:
        objects = _parse_voc_annotation(
            annotation_path,
            class_to_index=VOC_CLASS_TO_INDEX,
            skip_difficult=skip_difficult,
        )
        target_list.append(encode_yolov1_target(objects, s=s, b=b, c=c))
    y = cp.stack(target_list, axis=0).astype(cp.float32, copy=False)
    return x, y


def voc_image_target_batch(
    repo_root,
    batch_size=8,
    *,
    seed=0,
    batch_index=0,
    data_root=None,
    split="train",
    size=(448, 448),
    s=7,
    b=2,
    c=20,
    skip_difficult=True,
):
    """
    Return one Pascal VOC YOLOv1 batch: x (N,3,H,W) and y (N,S,S,B*5+C), both CuPy.
    """
    pairs = _voc_image_annotation_pairs(repo_root, data_root=data_root, split=split)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    start = batch_index * batch_size
    chunk = pairs[start : start + batch_size]
    if not chunk:
        raise ValueError(
            f"No samples for batch_index={batch_index} (dataset size {len(pairs)})"
        )
    return _voc_tensor_pair_from_pairs(
        chunk,
        size=size,
        s=s,
        b=b,
        c=c,
        skip_difficult=skip_difficult,
    )


def voc_dataset_size(repo_root, data_root=None, split="train") -> int:
    """Number of VOC image/annotation pairs in a split."""
    return len(_voc_image_annotation_pairs(repo_root, data_root=data_root, split=split))


def voc_num_batches_per_epoch(
    repo_root,
    batch_size,
    data_root=None,
    split="train",
) -> int:
    """Batches needed to cover VOC split once (last batch may be smaller)."""
    n = voc_dataset_size(repo_root, data_root=data_root, split=split)
    if n == 0:
        return 0
    return (n + batch_size - 1) // batch_size


# =============================================================================
# Fast VOC loader (module-level caches + threaded decode + CPU target encoding)
# =============================================================================

_PAIRS_CACHE: dict = {}
_ANNOS_CACHE: dict = {}
_PERM_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()

_EXECUTOR = ThreadPoolExecutor(max_workers=8)


def _cache_key(repo_root, data_root, split):
    return (str(Path(repo_root).resolve()) if repo_root is not None else None,
            str(Path(data_root).resolve()) if data_root is not None else None,
            str(split))


def _get_cached_pairs(repo_root, data_root, split):
    """Return cached list of (image_path, annotation_path) for (repo, data, split)."""
    key = _cache_key(repo_root, data_root, split)
    with _CACHE_LOCK:
        pairs = _PAIRS_CACHE.get(key)
    if pairs is not None:
        return key, pairs
    fresh_pairs = _voc_image_annotation_pairs(repo_root, data_root=data_root, split=split)
    with _CACHE_LOCK:
        _PAIRS_CACHE[key] = fresh_pairs
    return key, fresh_pairs


def _get_cached_annotations(key, pairs, skip_difficult=True):
    """Return dict image_id -> list of parsed object dicts, populated lazily."""
    with _CACHE_LOCK:
        cached = _ANNOS_CACHE.get(key)
    if cached is not None:
        return cached
    annos = {}
    for image_path, annotation_path in pairs:
        image_id = image_path.stem
        annos[image_id] = _parse_voc_annotation(
            annotation_path,
            class_to_index=VOC_CLASS_TO_INDEX,
            skip_difficult=skip_difficult,
        )
    with _CACHE_LOCK:
        _ANNOS_CACHE[key] = annos
    return annos


def _get_cached_permutation(key, seed, n):
    """Deterministic epoch-wise permutation of indices [0, n)."""
    pkey = (key, int(seed))
    with _CACHE_LOCK:
        perm = _PERM_CACHE.get(pkey)
    if perm is not None:
        return perm
    rng = random.Random(seed)
    fresh = list(range(n))
    rng.shuffle(fresh)
    with _CACHE_LOCK:
        _PERM_CACHE[pkey] = fresh
    return fresh


def _decode_one(path_and_size):
    path, size = path_and_size
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)  # CHW uint8


def _encode_yolov1_target_np(objects, s=7, b=2, c=20):
    """CPU (NumPy) equivalent of `encode_yolov1_target`. No GPU syncs."""
    target = np.zeros((s, s, b * 5 + c), dtype=np.float32)
    box_data_end = b * 5

    for obj in objects:
        x_center = float(obj["x_center"])
        y_center = float(obj["y_center"])
        box_width = float(obj["width"])
        box_height = float(obj["height"])
        class_index = int(obj["class_index"])

        if not (0.0 <= x_center < 1.0 and 0.0 <= y_center < 1.0):
            continue
        if box_width <= 0.0 or box_height <= 0.0:
            continue

        grid_x = min(int(x_center * s), s - 1)
        grid_y = min(int(y_center * s), s - 1)
        x_in_cell = x_center * s - grid_x
        y_in_cell = y_center * s - grid_y

        predictor_slot = None
        for predictor_index in range(b):
            confidence_channel = predictor_index * 5 + 4
            if target[grid_y, grid_x, confidence_channel] == 0.0:
                predictor_slot = predictor_index
                break
        if predictor_slot is None:
            continue

        base_channel = predictor_slot * 5
        target[grid_y, grid_x, base_channel + 0] = x_in_cell
        target[grid_y, grid_x, base_channel + 1] = y_in_cell
        target[grid_y, grid_x, base_channel + 2] = box_width
        target[grid_y, grid_x, base_channel + 3] = box_height
        target[grid_y, grid_x, base_channel + 4] = 1.0
        if 0 <= class_index < c:
            target[grid_y, grid_x, box_data_end + class_index] = 1.0

    return target


def _build_cpu_batch(
    repo_root,
    batch_size,
    *,
    seed,
    batch_index,
    data_root,
    split,
    size,
    s,
    b,
    c,
    skip_difficult=True,
    decode_workers=None,
):
    """
    Assemble one batch on the CPU:

    Returns:
        x_u8      : np.ndarray (N, 3, H, W) uint8, BGR->RGB handled (RGB here)
        y_np      : np.ndarray (N, S, S, B*5+C) float32
        objects   : list[list[dict]] of length N; unencoded objects per image
                    (used by GPU augmentation to recompute targets after warps)
    """
    key, pairs = _get_cached_pairs(repo_root, data_root, split)
    annos = _get_cached_annotations(key, pairs, skip_difficult=skip_difficult)
    n = len(pairs)
    perm = _get_cached_permutation(key, seed, n)

    start = batch_index * batch_size
    end = start + batch_size
    chunk_idx = perm[start:end]
    if not chunk_idx:
        raise ValueError(
            f"No samples for batch_index={batch_index} (dataset size {n})"
        )

    image_paths = [pairs[i][0] for i in chunk_idx]
    image_ids = [p.stem for p in image_paths]
    objects_per_image = [annos[image_id] for image_id in image_ids]

    decode_args = [(str(p), size) for p in image_paths]
    executor = _EXECUTOR if decode_workers is None else ThreadPoolExecutor(max_workers=decode_workers)
    try:
        chw_list = list(executor.map(_decode_one, decode_args))
    finally:
        if executor is not _EXECUTOR:
            executor.shutdown(wait=True)

    x_u8 = np.stack(chw_list, axis=0)

    y_np = np.stack(
        [_encode_yolov1_target_np(objs, s=s, b=b, c=c) for objs in objects_per_image],
        axis=0,
    )

    return x_u8, y_np, objects_per_image


# -----------------------------------------------------------------------------
# GPU augmentation (scale / translate / HSV jitter)
# -----------------------------------------------------------------------------


def _rgb_to_hsv_gpu(x):
    """x: (N, 3, H, W) float32 in [0,1] -> (h, s, v) each (N, H, W) float32."""
    r = x[:, 0]
    g = x[:, 1]
    bl = x[:, 2]
    mx = cp.maximum(cp.maximum(r, g), bl)
    mn = cp.minimum(cp.minimum(r, g), bl)
    df = mx - mn
    eps = cp.float32(1e-12)

    h = cp.zeros_like(mx)
    mask_df = df > 0
    mask_r = mask_df & (mx == r)
    mask_g = mask_df & (mx == g) & (~mask_r)
    mask_b = mask_df & (mx == bl) & (~mask_r) & (~mask_g)
    h = cp.where(mask_r, ((g - bl) / (df + eps)) % 6.0, h)
    h = cp.where(mask_g, ((bl - r) / (df + eps)) + 2.0, h)
    h = cp.where(mask_b, ((r - g) / (df + eps)) + 4.0, h)
    h = h / 6.0

    s = cp.where(mx > 0, df / (mx + eps), cp.zeros_like(mx))
    v = mx
    return h, s, v


def _hsv_to_rgb_gpu(h, s, v):
    """(h, s, v) each (N, H, W) in [0,1] -> (N, 3, H, W) float32."""
    h6 = h * 6.0
    i = cp.floor(h6).astype(cp.int32) % 6
    f = h6 - cp.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = cp.where(i == 0, v,
        cp.where(i == 1, q,
        cp.where(i == 2, p,
        cp.where(i == 3, p,
        cp.where(i == 4, t, v)))))
    g = cp.where(i == 0, t,
        cp.where(i == 1, v,
        cp.where(i == 2, v,
        cp.where(i == 3, q,
        cp.where(i == 4, p, p)))))
    b = cp.where(i == 0, p,
        cp.where(i == 1, p,
        cp.where(i == 2, t,
        cp.where(i == 3, v,
        cp.where(i == 4, v, q)))))
    return cp.stack([r, g, b], axis=1).astype(cp.float32)


def _augment_batch_gpu(
    x_float,
    objects_per_image,
    s,
    b,
    c,
    rng,
    scale_range=(0.8, 1.2),
    translate_range=(-0.2, 0.2),
    hsv_range=1.5,
):
    """
    Apply paper-style data augmentation on GPU.

    Scale/translate: per-image affine warp (bilinear) via cupyx.scipy.ndimage.affine_transform.
    HSV jitter: saturation and value (exposure) multiplied by factors in [1/hsv_range, hsv_range].

    Ground-truth boxes are warped by the same affine on the CPU and re-encoded via
    `_encode_yolov1_target_np`, then transferred once to the GPU.

    Args:
        x_float: (N, 3, H, W) float32 in [0, 1] on GPU.
        objects_per_image: list of length N of object dicts (unencoded).
        rng: np.random.Generator for deterministic per-batch augmentation.

    Returns:
        (x_aug, y_aug) on GPU.
    """
    from cupyx.scipy.ndimage import affine_transform

    n, _, h_img, w_img = x_float.shape

    scale = rng.uniform(scale_range[0], scale_range[1], size=n).astype(np.float32)
    tx = rng.uniform(translate_range[0], translate_range[1], size=n).astype(np.float32)
    ty = rng.uniform(translate_range[0], translate_range[1], size=n).astype(np.float32)
    sat = rng.uniform(1.0 / hsv_range, hsv_range, size=n).astype(np.float32)
    exp = rng.uniform(1.0 / hsv_range, hsv_range, size=n).astype(np.float32)

    x_warped = cp.empty_like(x_float)
    new_objects_per_image = [None] * n
    for i in range(n):
        scale_i = float(scale[i])
        tx_i = float(tx[i])
        ty_i = float(ty[i])
        inv_scale = 1.0 / scale_i

        matrix = cp.asarray(
            np.diag([1.0, inv_scale, inv_scale]).astype(np.float32)
        )
        off_y = h_img * 0.5 * (1.0 - inv_scale) - ty_i * h_img * inv_scale
        off_x = w_img * 0.5 * (1.0 - inv_scale) - tx_i * w_img * inv_scale
        offset = cp.asarray(np.array([0.0, off_y, off_x], dtype=np.float32))

        x_warped[i] = affine_transform(
            x_float[i], matrix, offset, order=1, mode="constant", cval=0.0
        )

        warped_objects = []
        for obj in objects_per_image[i]:
            cx = float(obj["x_center"])
            cy = float(obj["y_center"])
            w = float(obj["width"])
            h = float(obj["height"])

            cx_new = scale_i * (cx - 0.5) + 0.5 + tx_i
            cy_new = scale_i * (cy - 0.5) + 0.5 + ty_i
            w_new = scale_i * w
            h_new = scale_i * h

            if not (0.0 <= cx_new < 1.0 and 0.0 <= cy_new < 1.0):
                continue
            if w_new <= 0.0 or h_new <= 0.0:
                continue
            w_new = min(w_new, 1.0)
            h_new = min(h_new, 1.0)

            warped_objects.append({
                "class_index": obj["class_index"],
                "x_center": cx_new,
                "y_center": cy_new,
                "width": w_new,
                "height": h_new,
            })
        new_objects_per_image[i] = warped_objects

    h_ch, s_ch, v_ch = _rgb_to_hsv_gpu(x_warped)
    sat_gpu = cp.asarray(sat).reshape(n, 1, 1)
    exp_gpu = cp.asarray(exp).reshape(n, 1, 1)
    s_ch = cp.clip(s_ch * sat_gpu, 0.0, 1.0)
    v_ch = cp.clip(v_ch * exp_gpu, 0.0, 1.0)
    x_hsv = _hsv_to_rgb_gpu(h_ch, s_ch, v_ch)

    y_np = np.stack(
        [_encode_yolov1_target_np(objs, s=s, b=b, c=c) for objs in new_objects_per_image],
        axis=0,
    )
    y_gpu = cp.asarray(y_np)
    return x_hsv, y_gpu


def voc_image_target_batch_fast(
    repo_root,
    batch_size=8,
    *,
    seed=0,
    batch_index=0,
    data_root=None,
    split="train",
    size=(448, 448),
    s=7,
    b=2,
    c=20,
    skip_difficult=True,
    augment=False,
    aug_rng=None,
):
    """
    Drop-in replacement for `voc_image_target_batch` with caching, thread-pool
    decode, CPU target encoding, a single host->device transfer, and optional
    GPU augmentation.
    """
    x_u8, y_np, objects = _build_cpu_batch(
        repo_root, batch_size,
        seed=seed, batch_index=batch_index,
        data_root=data_root, split=split,
        size=size, s=s, b=b, c=c,
        skip_difficult=skip_difficult,
    )
    x = cp.asarray(x_u8).astype(cp.float32) / 255.0
    if augment:
        rng = aug_rng if aug_rng is not None else np.random.default_rng(
            (int(seed) * 1_000_003 + int(batch_index)) & 0xFFFFFFFF
        )
        x, y = _augment_batch_gpu(x, objects, s, b, c, rng)
    else:
        y = cp.asarray(y_np)
    return x, y


# -----------------------------------------------------------------------------
# Batch prefetcher (producer/consumer thread + queue)
# -----------------------------------------------------------------------------


class BatchPrefetcher:
    """
    Iterates batches for one epoch. A background thread runs ahead by up to
    ``max_prefetch`` batches, producing CPU-side (x_u8, y_np, objects) tuples.
    The main thread performs a single host->device transfer and optional GPU
    augmentation.

    CuPy calls happen only on the main thread so no CUDA context juggling is
    needed.
    """

    def __init__(
        self,
        repo_root,
        batch_size,
        *,
        seed,
        data_root,
        split,
        n_batches,
        size=(448, 448),
        s=7,
        b=2,
        c=20,
        augment=False,
        skip_difficult=True,
        max_prefetch=3,
        decode_workers=None,
    ):
        self._repo_root = repo_root
        self._data_root = data_root
        self._split = split
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._size = size
        self._s = int(s)
        self._b = int(b)
        self._c = int(c)
        self._augment = bool(augment)
        self._skip_difficult = bool(skip_difficult)
        self._n_batches = int(n_batches)
        self._decode_workers = decode_workers

        self._aug_rng = np.random.default_rng(self._seed * 1_000_003 + 7)

        self._q: queue.Queue = queue.Queue(maxsize=int(max_prefetch))
        self._stop = threading.Event()
        self._err = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            for batch_index in range(self._n_batches):
                if self._stop.is_set():
                    return
                cpu_payload = _build_cpu_batch(
                    self._repo_root, self._batch_size,
                    seed=self._seed, batch_index=batch_index,
                    data_root=self._data_root, split=self._split,
                    size=self._size, s=self._s, b=self._b, c=self._c,
                    skip_difficult=self._skip_difficult,
                    decode_workers=self._decode_workers,
                )
                while not self._stop.is_set():
                    try:
                        self._q.put(cpu_payload, timeout=0.25)
                        break
                    except queue.Full:
                        continue
            if not self._stop.is_set():
                self._q.put(None)
        except Exception as exc:
            self._err = exc
            try:
                self._q.put(None, timeout=1.0)
            except queue.Full:
                pass

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if self._err is not None:
            raise self._err
        if item is None:
            raise StopIteration
        x_u8, y_np, objects = item
        x = cp.asarray(x_u8).astype(cp.float32) / 255.0
        if self._augment:
            x, y = _augment_batch_gpu(x, objects, self._s, self._b, self._c, self._aug_rng)
        else:
            y = cp.asarray(y_np)
        return x, y

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)
