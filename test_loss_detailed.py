"""
test_loss_detailed.py

Term-by-term verification of yolo_loss using a hand-crafted single-cell
target.  Style mirrors the project notebooks: small explicit arrays, inline
derivations of expected values, cp.testing.assert_allclose for floats.

Run with:
    /usr/bin/python3 test_loss_detailed.py
"""

import sys
import types
import numpy as np

# ---------------------------------------------------------------------------
# CuPy shim: expose numpy as 'cupy' so loss.py imports unchanged on CPU.
# ---------------------------------------------------------------------------
cp_shim = types.ModuleType("cupy")
cp_shim.__dict__.update(np.__dict__)
cp_shim.asnumpy = np.array
cp_shim.asarray = np.asarray

_testing = types.ModuleType("cupy.testing")
_testing.assert_allclose = np.testing.assert_allclose
cp_shim.testing = _testing

sys.modules["cupy"] = cp_shim
sys.modules["cupy.testing"] = _testing

import numpy as cp                           # use numpy directly as cp alias
np.testing.assert_allclose                   # same handle
from loss import yolo_loss, yolo_loss_grad, _compute_assignments  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S, B, C  = 7, 2, 20
SLOT     = B * 5 + C   # 30
LC, LN   = 5.0, 0.5    # lambda_coord, lambda_noobj

# ---------------------------------------------------------------------------
# Canonical target: one GT object in cell (row=3, col=3), predictor slot 0.
#
#   x_cell = 0.5,  y_cell = 0.5  (centre of the cell)
#   w_img  = 0.3,  h_img  = 0.4  (relative to full image)
#   conf   = 1.0
#   class  = 0 (one-hot)
#   All other cells: zeros everywhere.
# ---------------------------------------------------------------------------
targets = np.zeros((1, S, S, SLOT), dtype=np.float32)
targets[0, 3, 3, 0] = 0.5          # x_cell  (slot 0)
targets[0, 3, 3, 1] = 0.5          # y_cell
targets[0, 3, 3, 2] = 0.3          # w_img
targets[0, 3, 3, 3] = 0.4          # h_img
targets[0, 3, 3, 4] = 1.0          # conf
targets[0, 3, 3, B*5 + 0] = 1.0    # class 0 one-hot

print("Target shape:", targets.shape)
print(f"GT cell (3,3) slot 0: {targets[0, 3, 3, :5]}")
print(f"GT cell (3,3) class:  {targets[0, 3, 3, B*5:]}")


# ---------------------------------------------------------------------------
# Helper: compute the five loss terms individually.
# ---------------------------------------------------------------------------
def compute_terms(preds_4d, tgt_4d):
    """Return (t1_xy, t2_wh, t3_obj, t4_noobj, t5_cls) as floats."""
    a   = _compute_assignments(preds_4d, tgt_4d, S, B)
    tc  = a['tgt_conf']           # (1,S,S,B)
    tb  = a['tgt_boxes']          # (1,S,S,B,4)
    tl  = a['tgt_cls']            # (1,S,S,C)
    rpb = a['resp_pred_boxes']    # (1,S,S,B,4)
    rpc = a['resp_pred_conf']     # (1,S,S,B)
    pc  = a['pred_conf']          # (1,S,S,B)
    pl  = a['pred_cls']           # (1,S,S,C)
    nij = a['noobj_ij']           # (1,S,S,B)
    om  = a['obj_cell']           # (1,S,S)

    dx = rpb[..., 0] - tb[..., 0]
    dy = rpb[..., 1] - tb[..., 1]
    t1 = LC * float((tc * (dx**2 + dy**2)).sum())

    pw = np.clip(rpb[..., 2], 1e-9, None)
    ph = np.clip(rpb[..., 3], 1e-9, None)
    gw = np.clip(tb[..., 2], 0.0, None)
    gh = np.clip(tb[..., 3], 0.0, None)
    t2 = LC * float((tc * ((np.sqrt(pw) - np.sqrt(gw))**2
                          + (np.sqrt(ph) - np.sqrt(gh))**2)).sum())

    t3 = float((tc * (rpc - 1.0)**2).sum())
    t4 = LN * float((nij * pc**2).sum())
    t5 = float((om[..., None] * (pl - tl)**2).sum())

    return t1, t2, t3, t4, t5


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def report(t1, t2, t3, t4, t5):
    total = t1 + t2 + t3 + t4 + t5
    print(f"  Term 1  xy coord  (λ={LC}):  {t1:>12.6f}")
    print(f"  Term 2  wh size   (λ={LC}):  {t2:>12.6f}")
    print(f"  Term 3  obj conf       :  {t3:>12.6f}")
    print(f"  Term 4  noobj conf(λ={LN}):  {t4:>12.6f}")
    print(f"  Term 5  class probs    :  {t5:>12.6f}")
    print(f"  {'─'*38}")
    print(f"  Total                  :  {total:>12.6f}")


# ===========================================================================
# TEST A — Perfect predictions (pred == target)
# ===========================================================================
section("TEST A — Perfect predictions (pred == target)")

perf = targets.copy()
print("\nInput: predictions copied verbatim from targets.")

t1, t2, t3, t4, t5 = compute_terms(perf, targets)
report(t1, t2, t3, t4, t5)

# Every single term must be exactly 0.
# Derivation for each:
#
#   Term 1: dx = pred_x - gt_x = 0,  dy = 0  →  0
#   Term 2: pred_w = gt_w = 0.3  →  √0.3 - √0.3 = 0  →  0
#   Term 3: responsible pred conf = 1.0,  target conf = 1.0  →  (1-1)² = 0
#   Term 4: all non-responsible confs = 0  →  0² = 0
#   Term 5: pred_cls = gt_cls (same one-hot)  →  0
print()
for name, val in [("xy coord", t1), ("wh size", t2), ("obj conf", t3),
                  ("noobj conf", t4), ("class probs", t5)]:
    assert val == 0.0, f"[BUG] {name} term should be 0.0 but got {val:.8f}"
    print(f"  [PASS] {name} term = 0.0  (pred == target → diff = 0)")


# ===========================================================================
# TEST B — Slightly wrong coordinates (noise only on GT-cell coords)
# ===========================================================================
section("TEST B — Noisy coordinates (+0.05 to x,y,w,h in cell (3,3))")

NOISE = 0.05
noisy = targets.copy()
noisy[0, 3, 3, 0] += NOISE    # pred_x = 0.55  (gt_x = 0.50)
noisy[0, 3, 3, 1] += NOISE    # pred_y = 0.55  (gt_y = 0.50)
noisy[0, 3, 3, 2] += NOISE    # pred_w = 0.35  (gt_w = 0.30)
noisy[0, 3, 3, 3] += NOISE    # pred_h = 0.45  (gt_h = 0.40)
# conf and class channels are identical to target → terms 3,4,5 stay 0

print(f"\n  pred box in (3,3): {noisy[0, 3, 3, :5]}")
print(f"  gt   box in (3,3): {targets[0, 3, 3, :5]}")

t1n, t2n, t3n, t4n, t5n = compute_terms(noisy, targets)
report(t1n, t2n, t3n, t4n, t5n)

# ---- Term 1 hand derivation ----
# dx = 0.55 - 0.50 = 0.05,  dy = 0.55 - 0.50 = 0.05
# t1 = λ_coord * (dx² + dy²) = 5 * (0.05² + 0.05²) = 5 * 0.005 = 0.025
expected_t1 = LC * (NOISE**2 + NOISE**2)

# ---- Term 2 hand derivation ----
# pred_w = 0.35,  gt_w = 0.30  →  √0.35 - √0.30
# pred_h = 0.45,  gt_h = 0.40  →  √0.45 - √0.40
pw_exp = 0.30 + NOISE
ph_exp = 0.40 + NOISE
dw_exp = np.sqrt(pw_exp) - np.sqrt(0.30)
dh_exp = np.sqrt(ph_exp) - np.sqrt(0.40)
expected_t2 = LC * (dw_exp**2 + dh_exp**2)

print(f"\n  Hand-computed expected values:")
print(f"  Term 1: λ*(dx²+dy²) = {LC}*({NOISE}²+{NOISE}²) = {expected_t1:.6f}")
print(f"  Term 2: λ*((√{pw_exp:.2f}-√0.30)²+(√{ph_exp:.2f}-√0.40)²)")
print(f"        = {LC}*({dw_exp:.6f}²+{dh_exp:.6f}²) = {expected_t2:.6f}")

np.testing.assert_allclose(t1n, expected_t1, rtol=1e-5,
    err_msg=f"Term 1 mismatch: got {t1n:.6f}, expected {expected_t1:.6f}")
np.testing.assert_allclose(t2n, expected_t2, rtol=1e-5,
    err_msg=f"Term 2 mismatch: got {t2n:.6f}, expected {expected_t2:.6f}")

# Conf and class channels are still == target → terms 3,4,5 must be 0
assert t3n == 0.0, f"[BUG] obj conf term should be 0 but got {t3n:.8f}"
assert t4n == 0.0, f"[BUG] noobj term should be 0 but got {t4n:.8f}"
assert t5n == 0.0, f"[BUG] class term should be 0 but got {t5n:.8f}"

print()
print(f"  [PASS] Term 1 (xy) increased to {t1n:.6f}  (expected {expected_t1:.6f})")
print(f"  [PASS] Term 2 (wh) increased to {t2n:.6f}  (expected {expected_t2:.6f})")
print(f"  [PASS] Term 3 (obj conf) unchanged: {t3n:.6f}  (conf channel not touched)")
print(f"  [PASS] Term 4 (noobj) unchanged: {t4n:.6f}  (noobj cells still predict 0)")
print(f"  [PASS] Term 5 (class) unchanged: {t5n:.6f}  (class channel not touched)")
print(f"  [PASS] λ_coord = {LC} amplification verified for both xy and wh terms")


# ===========================================================================
# TEST C — Wrong confidence only
#   • pred_conf = 0.0  in GT cell (3,3) slot 0  → obj loss fires
#   • pred_conf = 0.5  everywhere else           → noobj loss fires
#   • coords and class identical to target       → terms 1, 2, 5 stay 0
# ===========================================================================
section("TEST C — Wrong confidence only")
print("  pred_conf = 0.0 in GT cell (3,3) slot 0  (should be 1.0)")
print("  pred_conf = 0.5 everywhere else           (should be 0.0)")
print("  coords and class copied from target        (unchanged)")

conf_wrong = targets.copy()
# Set all confidence channels to 0.5 (wrong for noobj cells)
for b in range(B):
    conf_wrong[0, :, :, b*5 + 4] = 0.5
# Override: GT cell responsible predictor gets 0 instead of 1
conf_wrong[0, 3, 3, 0*5 + 4] = 0.0    # slot 0 conf  →  0  (wrong)
conf_wrong[0, 3, 3, 1*5 + 4] = 0.5    # slot 1 conf  →  0.5 (same as rest)

print(f"\n  pred conf in (3,3): slot0={conf_wrong[0,3,3,4]:.1f}"
      f"  slot1={conf_wrong[0,3,3,9]:.1f}")

t1c, t2c, t3c, t4c, t5c = compute_terms(conf_wrong, targets)
report(t1c, t2c, t3c, t4c, t5c)

# ---- Coords/class unchanged → terms 1, 2, 5 must be 0 ----
assert t1c == 0.0, f"[BUG] xy term should be 0, got {t1c:.8f}"
assert t2c == 0.0, f"[BUG] wh term should be 0, got {t2c:.8f}"
assert t5c == 0.0, f"[BUG] class term should be 0, got {t5c:.8f}"

# ---- Term 3: obj confidence fires in cell (3,3) ----
#   responsible predictor for GT slot 0 = pred slot 0
#   (pred slot 0 has the same geometry as GT → IoU=1.0 > IoU(slot 1, GT)≈0)
#   t3 = 1 GT box * (pred_conf - 1)² = (0.0 - 1.0)² = 1.0
expected_t3 = 1.0 * (0.0 - 1.0)**2

np.testing.assert_allclose(t3c, expected_t3, rtol=1e-5,
    err_msg=f"Term 3 mismatch: got {t3c:.6f}, expected {expected_t3:.6f}")

# ---- Term 4: noobj confidence fires everywhere non-responsible ----
#   obj_ij  = 1 only at (3,3, pred_slot_0)   → 1 pair excluded
#   noobj_ij = 1 for the remaining:
#     • cell (3,3), slot 1                  →  1 pair
#     • all other 48 cells, both B=2 slots  →  48 * 2 = 96 pairs
#     Total: 97 pairs, each with pred_conf = 0.5
#   t4 = λ_noobj * 97 * 0.5² = 0.5 * 97 * 0.25 = 12.125
n_noobj_pairs = (S * S - 1) * B + (B - 1)   # = 96 + 1 = 97
expected_t4   = LN * n_noobj_pairs * (0.5**2)

np.testing.assert_allclose(t4c, expected_t4, rtol=1e-5,
    err_msg=f"Term 4 mismatch: got {t4c:.6f}, expected {expected_t4:.6f}")

print()
print(f"  [PASS] Term 1 (xy) = 0       — coords not touched")
print(f"  [PASS] Term 2 (wh) = 0       — sizes not touched")
print(f"  [PASS] Term 3 (obj conf) = {t3c:.4f}")
print(f"         derivation: 1 GT box × (pred_conf=0 − 1)² = {expected_t3:.4f}")
print(f"  [PASS] Term 4 (noobj conf) = {t4c:.4f}")
print(f"         derivation: λ={LN} × {n_noobj_pairs} pairs × 0.5² = {expected_t4:.4f}")
print(f"         ({n_noobj_pairs} = 48 other cells × 2 slots + slot 1 in GT cell)")
print(f"  [PASS] Term 5 (class) = 0    — class channel not touched")

# ---- Cross-check: sum of 5 terms == yolo_loss scalar ----
expected_total = t1c + t2c + t3c + t4c + t5c
actual_total   = yolo_loss(conf_wrong.reshape(1, -1), targets,
                           S=S, B=B, C=C,
                           lambda_coord=LC, lambda_noobj=LN)
np.testing.assert_allclose(actual_total, expected_total, rtol=1e-5,
    err_msg=f"yolo_loss {actual_total:.6f} != sum of terms {expected_total:.6f}")
print(f"\n  [PASS] yolo_loss() total ({actual_total:.4f}) == sum of 5 terms ({expected_total:.4f})")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print("All tests passed.")
print("=" * 60)
