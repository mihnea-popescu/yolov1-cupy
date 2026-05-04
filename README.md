# YOLOv1 — from scratch in CuPy

A complete implementation of [You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) built from mathematical first principles using CuPy. No PyTorch, no TensorFlow — every layer, loss term, and gradient is hand-written and runs on GPU via CUDA.

---

## Overview

This project re-implements the full YOLOv1 pipeline:

- **Neural network layers** written from scratch: convolution (im2col), batch normalization, max/avg pooling, dropout, leaky ReLU, fully-connected, softmax
- **Darknet-19 backbone** with support for loading original Darknet binary weights (`.weights` format)
- **YOLOv1 detection head** producing 7×7×30 output tensors (2 boxes + 20 VOC classes per cell)
- **Multi-part YOLO loss** with IoU-based responsible-predictor assignment
- **Pascal VOC 2012** data loader with bounding-box → grid encoding
- **SGD with momentum** and a configurable LR schedule

The goal was to understand every moving part of a real object detector — from `im2col` in the convolution backward pass to the `sqrt(w·h)` gradient clipping in the loss.

---

## Architecture

### YOLOv1 (`yolo.py`)

| Stage | Details |
|---|---|
| Input | (N, 3, 448, 448) |
| Backbone | 24 conv blocks, leaky ReLU, batch norm, max pooling |
| Detection head | 2 conv layers → 2 stride-2 reductions → flatten → FC 4096 → FC 1470 |
| Output | (N, 7, 7, 30) — 2 boxes × 5 + 20 class scores per cell |

### Darknet-19 backbone (`darknet.py`)

16 conv blocks with channel progression 64 → 192 → 512 → 1024, global average pooling, and a classification head. Used standalone for ImageNet pre-training and as a feature extractor in YOLO.

### Mini-Darknet (`mini_darknet.py`)

A lightweight 5-block variant (16 → 32 → 64 → 128 → 256 channels) for fast iteration and layer-level experiments.

---

## Features

- **No framework dependencies** — forward and backward passes implemented in CuPy/NumPy
- **Darknet weight loading** — parses the original binary `.weights` format, including conv + BN fused blocks
- **IoU-based box assignment** — computes the full (N, S, S, B_pred, B_gt) IoU matrix to find the responsible predictor per cell
- **Softmax + cross-entropy class loss** — deviation from the paper's MSE, giving better-conditioned gradients for rare classes
- **Numerically stable** throughout: max-subtraction in softmax, `sqrt` gradient clipping, `im2col` via stride tricks
- **Train/eval mode** — batch norm switches between mini-batch stats (train) and running stats (eval); inverted dropout

---

## Project Structure

```
yolov1-cupy/
├── yolo.py                   # YOLOv1 full model
├── darknet.py                # Darknet-19 backbone + classifier
├── mini_darknet.py           # Lightweight backbone variant
├── loss.py                   # YOLOv1 multi-part loss + gradients
├── image_batch_loader.py     # Pascal VOC 2012 + generic image loader
├── conv2d.py                 # Conv2D (im2col, forward + backward)
├── batchnorm2d.py            # Batch normalization
├── linear.py                 # Fully-connected layer
├── leaky_relu.py             # Leaky ReLU activation
├── maxpool.py                # 2D max pooling
├── avgpool2d.py              # 2D average pooling
├── global_avg_pool2d.py      # Global average pooling
├── dropout.py                # Inverted dropout
├── flatten.py                # Flatten layer
├── softmax.py                # Numerically stable softmax
├── cross_entropy.py          # Cross-entropy loss
├── test_loss.py              # Loss unit tests (CPU, numpy shim)
├── test_loss_detailed.py     # Gradient validation for loss terms
├── models/                   # Saved checkpoints (.npz)
└── notebooks/
    ├── YOLO_train_v2.ipynb   # Main training notebook (VOC 2012)
    ├── YOLO_test.ipynb       # Inference and evaluation
    ├── YOLO_backbone_insertion.ipynb  # Backbone swap experiments
    └── ...                   # Per-layer validation notebooks
```

---

## Requirements

```
cupy-cudaXXX   # match your CUDA version, e.g. cupy-cuda12x
numpy
pillow
```

A `.venv/` with the required packages is included. Activate it with:

```bash
source .venv/bin/activate
```

---

## Dataset

Download Pascal VOC 2012 train/val and place it so the path contains `JPEGImages/` and `Annotations/`:

```
yolov1-cupy/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── Annotations/
        └── ImageSets/
```

The loader also accepts `pascal_voc_2012/` or `VOC2012_train_val/VOC2012/` at the repo root.

---

## Training

Open `notebooks/YOLO_train_v2.ipynb`. The key loop:

```python
import cupy as cp
from yolo import YOLO
from loss import yolo_loss, yolo_loss_grad
from image_batch_loader import voc_image_target_batch_fast

S, B, C = 7, 2, 20
model = YOLO(num_classes=C, dtype=cp.float32)
model.load_backbone_weights("models/darknet_pretrained-epoch15.npz")
model.train()

for epoch in range(60):
    lr = get_lr(epoch)   # warmup → 1e-2 → 1e-3 → 1e-4
    x, y = voc_image_target_batch_fast(
        REPO, batch_size=64, seed=epoch,
        data_root=VOC_DATA_ROOT, split="train",
        size=(448, 448), s=S, b=B, c=C, augment=True,
    )
    logits = model.forward(x)
    loss   = yolo_loss(logits, y, S=S, B=B, C=C, lambda_coord=5.0, lambda_noobj=0.1)
    grad   = yolo_loss_grad(logits, y, S=S, B=B, C=C, lambda_coord=5.0, lambda_noobj=0.1)
    model.backward(grad)
    model.sgd_momentum_step(lr, momentum=0.9, weight_decay=5e-4)
    model.zero_grad()
```

---

## Evaluation

Open `notebooks/YOLO_test.ipynb`:

```python
model.eval()
logits = model.forward(x_test)   # (N, 7, 7, 30)
# decode → boxes, scores, class ids
# apply NMS per class
```

---

## Results

### Darknet-19 on ImageNet-10

Pre-training the backbone on a 10-class ImageNet subset before fine-tuning on VOC:

| Epoch | Train loss | Val loss | Val accuracy |
|------:|----------:|--------:|------------:|
| 1 | 1.54 | 1.28 | 55.8% |
| 5 | 0.53 | 0.64 | 78.3% |
| 10 | 0.19 | 0.55 | 83.5% |
| 15 | 0.09 | 0.56 | **85.1%** |

### YOLOv1 on Pascal VOC 2012

End-to-end training with the pretrained backbone converges but exhibits **class-head collapse** — the model predicts `person` for 100% of detections. Known causes and mitigations:

- VOC class imbalance (`person` in ~40% of images) with no per-class loss weighting
- Large untrained head (249M params) on a small dataset (5,717 images)
- No backbone freezing during the early epochs

Planned fixes: freeze backbone for first 10 epochs, add inverse-frequency class weights, reduce initial LR.

---

## Implementation Notes

**Convolution via im2col** (`conv2d.py`) — input patches are unrolled into columns using `stride_tricks`, turning the sliding-window operation into a single matrix multiply. The backward pass uses `cupyx.scatter_add` for the col2im accumulation.

**Responsible predictor assignment** (`loss.py`) — for each ground-truth box in a cell, the predicted box with the highest IoU is designated "responsible". The IoU computation is treated as a stop-gradient operation; only the responsible box's coordinate and confidence losses are back-propagated.

**sqrt(w·h) gradient** — the paper scales width/height loss by `sqrt`, which introduces a `1 / (2·sqrt(w))` term in the gradient. Predictions are clamped to `max(w, 1e-9)` to prevent blow-up at initialization when predicted widths can be near zero or negative.

**Softmax-CE vs MSE for class loss** — the original paper uses sum-of-squared error on raw class probabilities, which gives weak, uniform gradients. This implementation uses softmax + cross-entropy, matching common modern reproductions and providing sharper per-class gradients.
