import random
from pathlib import Path
import xml.etree.ElementTree as ET

import cupy as cp
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
