"""
YOLOv1 loss function — Redmon et al. (2016), Equation 3.

Tensor convention (matches this codebase):
  predictions : (N, S*S*(B*5+C)) flat — raw output of YOLO.forward(), or
                (N, S, S, B*5+C)       — pre-reshaped
  targets     : (N, S, S, B*5+C)       — from encode_yolov1_target()

Per-predictor slot layout (both pred and target):
  channels  [b*5 + 0]  : x_center relative to cell  [0, 1)
  channels  [b*5 + 1]  : y_center relative to cell  [0, 1)
  channels  [b*5 + 2]  : width  relative to full image [0, 1]
  channels  [b*5 + 3]  : height relative to full image [0, 1]
  channels  [b*5 + 4]  : objectness confidence
  channels  [B*5 : ]   : class one-hot vector (C classes)

Responsible-predictor rule (Section 2.2 of the paper):
  For each GT slot b_gt with a ground-truth object, the predicted box whose
  image-space IoU with that GT box is highest is deemed "responsible".
  The IoU assignment is treated as a stop-gradient operation (straight-through).
"""

import cupy as cp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_components(tensor: cp.ndarray, B: int):
    """Split (..., B*5+C) tensor into boxes, confidences, and class probs."""
    boxes = cp.stack([tensor[..., b * 5: b * 5 + 4] for b in range(B)], axis=-2)
    confs = cp.stack([tensor[..., b * 5 + 4]        for b in range(B)], axis=-1)
    cls   = tensor[..., B * 5:]
    return boxes, confs, cls


def _to_image_corners(boxes: cp.ndarray, S: int) -> cp.ndarray:
    """
    Convert (N, S, S, B, 4) boxes in (cx_cell, cy_cell, w_img, h_img) format
    to image-space corners (x1, y1, x2, y2).

    cx_cell / cy_cell are relative to the cell origin [0, 1).
    w_img   / h_img   are relative to the full image  [0, 1].
    """
    # Cell-origin offsets — col index for x, row index for y
    col_off = cp.arange(S, dtype=cp.float32).reshape(1, 1, S, 1)  # (1,1,S,1)
    row_off = cp.arange(S, dtype=cp.float32).reshape(1, S, 1, 1)  # (1,S,1,1)

    cx = (col_off + boxes[..., 0]) / S          # (N,S,S,B)
    cy = (row_off + boxes[..., 1]) / S
    w  = cp.clip(boxes[..., 2], 0.0, None)
    h  = cp.clip(boxes[..., 3], 0.0, None)

    return cp.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def _compute_assignments(
    preds:   cp.ndarray,   # (N, S, S, B*5+C)
    targets: cp.ndarray,   # (N, S, S, B*5+C)
    S: int,
    B: int,
) -> dict:
    """
    Compute all intermediate quantities shared by the loss and gradient:
      - extracted box / conf / class tensors
      - cross-IoU matrix between every predicted box and every GT box
      - responsible predicted-box index for each GT slot
      - obj / noobj indicator masks
    """
    N = preds.shape[0]

    pred_boxes, pred_conf, pred_cls = _extract_components(preds,   B)
    tgt_boxes,  tgt_conf,  tgt_cls  = _extract_components(targets, B)

    # ---- image-space corners for IoU ----------------------------------------
    pred_corners = _to_image_corners(pred_boxes, S)   # (N,S,S,B_pred,4)
    tgt_corners  = _to_image_corners(tgt_boxes,  S)   # (N,S,S,B_gt, 4)

    # ---- cross-IoU: pred b_pred vs GT b_gt ----------------------------------
    pc = pred_corners[:, :, :, :, None, :]   # (N,S,S,B_pred,1,    4)
    tc = tgt_corners[ :, :, :, None, :, :]   # (N,S,S,1,    B_gt,  4)

    ix1 = cp.maximum(pc[..., 0], tc[..., 0])
    iy1 = cp.maximum(pc[..., 1], tc[..., 1])
    ix2 = cp.minimum(pc[..., 2], tc[..., 2])
    iy2 = cp.minimum(pc[..., 3], tc[..., 3])
    inter = cp.maximum(ix2 - ix1, 0.0) * cp.maximum(iy2 - iy1, 0.0)

    pred_area = (pc[..., 2] - pc[..., 0]) * (pc[..., 3] - pc[..., 1])  # (N,S,S,B,1)
    tgt_area  = (tc[..., 2] - tc[..., 0]) * (tc[..., 3] - tc[..., 1])  # (N,S,S,1,B)
    union = pred_area + tgt_area - inter
    iou   = inter / cp.maximum(union, 1e-6)   # (N,S,S,B_pred,B_gt)

    # For each GT slot b_gt, pick the predicted box with the highest IoU.
    # argmax over the B_pred axis → (N,S,S,B_gt)
    best_pred = iou.argmax(axis=3)

    # ---- gather responsible predicted-box tensors ---------------------------
    n_idx = cp.arange(N)[:, None, None, None]
    i_idx = cp.arange(S)[None, :, None, None]
    j_idx = cp.arange(S)[None, None, :, None]

    # resp_pred_boxes[n,i,j,b_gt,:] = pred_boxes[n,i,j,best_pred[n,i,j,b_gt],:]
    resp_pred_boxes = pred_boxes[n_idx, i_idx, j_idx, best_pred, :]  # (N,S,S,B_gt,4)
    resp_pred_conf  = pred_conf[ n_idx, i_idx, j_idx, best_pred   ]  # (N,S,S,B_gt)

    # ---- obj_ij mask: 1 where pred box b_pred is responsible for some GT ----
    # Iterate over B*B combinations (B=2, so 4 iterations — negligible cost).
    obj_ij = cp.zeros((N, S, S, B), dtype=cp.float32)
    for b_gt in range(B):
        gt_has  = tgt_conf[..., b_gt]                               # (N,S,S)
        for b_pred in range(B):
            is_resp = (best_pred[..., b_gt] == b_pred).astype(cp.float32)
            obj_ij[..., b_pred] += gt_has * is_resp

    obj_ij  = cp.clip(obj_ij, 0.0, 1.0)
    noobj_ij = 1.0 - obj_ij

    # ---- cell-level object mask (for class loss) ----------------------------
    obj_cell = (tgt_conf.max(axis=-1) > 0.5).astype(cp.float32)    # (N,S,S)

    return dict(
        pred_boxes=pred_boxes, pred_conf=pred_conf, pred_cls=pred_cls,
        tgt_boxes=tgt_boxes,   tgt_conf=tgt_conf,   tgt_cls=tgt_cls,
        resp_pred_boxes=resp_pred_boxes, resp_pred_conf=resp_pred_conf,
        best_pred=best_pred,
        obj_ij=obj_ij, noobj_ij=noobj_ij, obj_cell=obj_cell,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def yolo_loss(
    predictions: cp.ndarray,
    targets:     cp.ndarray,
    S:             int   = 7,
    B:             int   = 2,
    C:             int   = 20,
    lambda_coord:  float = 5.0,
    lambda_noobj:  float = 0.5,
) -> float:
    """
    YOLOv1 multi-part loss (Equation 3 of Redmon et al. 2016).

    Parameters
    ----------
    predictions : cp.ndarray, shape (N, S*S*(B*5+C)) or (N, S, S, B*5+C)
        Raw output of YOLO.forward().
    targets : cp.ndarray, shape (N, S, S, B*5+C)
        Ground-truth tensor produced by encode_yolov1_target().
    S : int
        Grid size (default 7).
    B : int
        Number of bounding-box predictors per cell (default 2).
    C : int
        Number of classes (default 20 for VOC).
    lambda_coord : float
        Weight for coordinate loss (default 5).
    lambda_noobj : float
        Weight for no-object confidence loss (default 0.5).

    Returns
    -------
    float
        Scalar loss value (sum over the batch, matching Eq. 3).
    """
    N = predictions.shape[0]
    preds = predictions.reshape(N, S, S, B * 5 + C)

    a = _compute_assignments(preds, targets, S, B)

    tgt_conf        = a['tgt_conf']          # (N,S,S,B)
    tgt_boxes       = a['tgt_boxes']         # (N,S,S,B,4)
    tgt_cls         = a['tgt_cls']           # (N,S,S,C)
    resp_pred_boxes = a['resp_pred_boxes']   # (N,S,S,B,4)
    resp_pred_conf  = a['resp_pred_conf']    # (N,S,S,B)
    pred_conf       = a['pred_conf']         # (N,S,S,B)
    pred_cls        = a['pred_cls']          # (N,S,S,C)
    noobj_ij        = a['noobj_ij']          # (N,S,S,B)
    obj_cell        = a['obj_cell']          # (N,S,S)

    # ------------------------------------------------------------------
    # Term 1: xy coordinate loss   λ_coord Σ 1^obj_ij [(x-x̂)²+(y-ŷ)²]
    # ------------------------------------------------------------------
    dx = resp_pred_boxes[..., 0] - tgt_boxes[..., 0]   # (N,S,S,B)
    dy = resp_pred_boxes[..., 1] - tgt_boxes[..., 1]
    loss_xy = (tgt_conf * (dx ** 2 + dy ** 2)).sum()

    # ------------------------------------------------------------------
    # Term 2: wh coordinate loss   λ_coord Σ 1^obj_ij [(√w-√ŵ)²+(√h-√ĥ)²]
    # ------------------------------------------------------------------
    pw = cp.clip(resp_pred_boxes[..., 2], 1e-9, None)
    ph = cp.clip(resp_pred_boxes[..., 3], 1e-9, None)
    gw = cp.clip(tgt_boxes[..., 2], 0.0, None)
    gh = cp.clip(tgt_boxes[..., 3], 0.0, None)
    loss_wh = (tgt_conf * ((cp.sqrt(pw) - cp.sqrt(gw)) ** 2
                          + (cp.sqrt(ph) - cp.sqrt(gh)) ** 2)).sum()

    loss_coord = lambda_coord * (loss_xy + loss_wh)

    # ------------------------------------------------------------------
    # Term 3: object confidence loss   Σ 1^obj_ij (C_i - Ĉ_i)²
    # Target confidence = 1.0 for the responsible predictor.
    # ------------------------------------------------------------------
    loss_obj = (tgt_conf * (resp_pred_conf - 1.0) ** 2).sum()

    # ------------------------------------------------------------------
    # Term 4: no-object confidence loss   λ_noobj Σ 1^noobj_ij (C_i - Ĉ_i)²
    # Target confidence = 0.0 for all non-responsible predictors.
    # ------------------------------------------------------------------
    loss_noobj = lambda_noobj * (noobj_ij * pred_conf ** 2).sum()

    # ------------------------------------------------------------------
    # Term 5: class probability loss   Σ 1^obj_i Σ_c (p_i(c) - p̂_i(c))²
    # ------------------------------------------------------------------
    loss_cls = (obj_cell[..., None] * (pred_cls - tgt_cls) ** 2).sum()

    total = loss_coord + loss_obj + loss_noobj + loss_cls
    return float(cp.asnumpy(total))


def yolo_loss_grad(
    predictions: cp.ndarray,
    targets:     cp.ndarray,
    S:             int   = 7,
    B:             int   = 2,
    C:             int   = 20,
    lambda_coord:  float = 5.0,
    lambda_noobj:  float = 0.5,
) -> cp.ndarray:
    """
    Gradient of the YOLOv1 loss w.r.t. ``predictions``.

    The IoU-based responsible-predictor assignment is treated as a
    stop-gradient (straight-through), so the gradient flows only through
    the loss terms, not through the IoU computation.

    Parameters
    ----------
    predictions : cp.ndarray
        Same shape as supplied to yolo_loss().
    targets : cp.ndarray
        Same as supplied to yolo_loss().

    Returns
    -------
    cp.ndarray
        dL/d(predictions), same shape as ``predictions``.
    """
    N = predictions.shape[0]
    input_shape = predictions.shape
    preds = predictions.reshape(N, S, S, B * 5 + C)

    a = _compute_assignments(preds, targets, S, B)

    tgt_conf  = a['tgt_conf']    # (N,S,S,B)
    tgt_boxes = a['tgt_boxes']   # (N,S,S,B,4)
    tgt_cls   = a['tgt_cls']     # (N,S,S,C)
    pred_cls  = a['pred_cls']    # (N,S,S,C)
    noobj_ij  = a['noobj_ij']    # (N,S,S,B)
    obj_cell  = a['obj_cell']    # (N,S,S)
    best_pred = a['best_pred']   # (N,S,S,B_gt) — responsible pred index per GT slot

    grad = cp.zeros_like(preds)  # (N,S,S,B*5+C)

    # ------------------------------------------------------------------
    # Gradients from coordinate and obj-confidence losses.
    # Route through the responsible predictor for each GT slot.
    # ------------------------------------------------------------------
    for b_gt in range(B):
        m = tgt_conf[..., b_gt]         # (N,S,S) — 1 where GT exists at slot b_gt

        gt_x = tgt_boxes[..., b_gt, 0]  # (N,S,S)
        gt_y = tgt_boxes[..., b_gt, 1]
        gt_w = cp.clip(tgt_boxes[..., b_gt, 2], 0.0, None)
        gt_h = cp.clip(tgt_boxes[..., b_gt, 3], 0.0, None)

        for b_pred in range(B):
            # active[n,i,j] = 1 iff pred b_pred is responsible for GT slot b_gt
            # AND that GT slot actually has an object.
            is_resp = (best_pred[..., b_gt] == b_pred).astype(cp.float32)
            active  = m * is_resp               # (N,S,S)

            base = b_pred * 5
            px = preds[..., base + 0]           # (N,S,S)
            py = preds[..., base + 1]
            pw_raw = preds[..., base + 2]
            ph_raw = preds[..., base + 3]
            pc = preds[..., base + 4]

            # dL/d(x) = 2 λ_coord * active * (pred_x - gt_x)
            grad[..., base + 0] += 2.0 * lambda_coord * active * (px - gt_x)
            grad[..., base + 1] += 2.0 * lambda_coord * active * (py - gt_y)

            # dL/d(w) via chain rule through sqrt(clip(w, eps, inf)):
            #   d/dw [ (√w - √ĝw)² ] = (√w - √ĝw)/√w   if w > eps
            #                       = 0                 if w <= eps (clipped branch)
            # The old code used sqrt(max(w, eps)) in the denominator, which makes
            # the gradient diverge (~ -√gt_w / 6e-5) whenever the raw prediction
            # is negative -- a catastrophic blow-up on randomly-initialised heads.
            pw_clip_eps = 1e-9
            ph_clip_eps = 1e-9
            w_valid = (pw_raw > pw_clip_eps).astype(preds.dtype)
            h_valid = (ph_raw > ph_clip_eps).astype(preds.dtype)
            pw = cp.maximum(pw_raw, pw_clip_eps)
            ph = cp.maximum(ph_raw, ph_clip_eps)
            grad[..., base + 2] += (
                2.0 * lambda_coord * active * w_valid
                * (cp.sqrt(pw) - cp.sqrt(gt_w)) / (2.0 * cp.sqrt(pw))
            )
            grad[..., base + 3] += (
                2.0 * lambda_coord * active * h_valid
                * (cp.sqrt(ph) - cp.sqrt(gt_h)) / (2.0 * cp.sqrt(ph))
            )

            # dL/d(conf) from obj loss: 2 * active * (pred_conf - 1)
            grad[..., base + 4] += 2.0 * active * (pc - 1.0)

    # ------------------------------------------------------------------
    # Gradient from no-object confidence loss.
    # dL/d(pred_conf_b) = 2 λ_noobj * noobj_ij_b * pred_conf_b
    # (target confidence = 0, so loss = pred_conf²)
    # ------------------------------------------------------------------
    for b_pred in range(B):
        base = b_pred * 5
        pc = preds[..., base + 4]
        grad[..., base + 4] += 2.0 * lambda_noobj * noobj_ij[..., b_pred] * pc

    # ------------------------------------------------------------------
    # Gradient from class probability loss.
    # dL/d(pred_cls_c) = 2 * obj_cell * (pred_cls_c - gt_cls_c)
    # ------------------------------------------------------------------
    grad[..., B * 5:] += 2.0 * obj_cell[..., None] * (pred_cls - tgt_cls)

    return grad.reshape(input_shape)
