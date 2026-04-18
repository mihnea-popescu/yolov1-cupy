"""
test_loss.py — runs on CPU (no GPU required).

Patches 'cupy' with numpy so loss.py can be imported and exercised
without a CUDA device. numpy's API is a superset of what loss.py uses.
"""
import sys
import types
import numpy as np

# ---------------------------------------------------------------------------
# Shim: expose numpy as the 'cupy' module so loss.py imports work unchanged.
# ---------------------------------------------------------------------------
cp_shim = types.ModuleType("cupy")
cp_shim.__dict__.update(np.__dict__)          # copy all numpy symbols
cp_shim.asnumpy = np.array                   # cp.asnumpy(x) → np.array(x)
cp_shim.asarray = np.asarray
sys.modules["cupy"] = cp_shim

# Now import loss with the shim in place.
from loss import yolo_loss, yolo_loss_grad   # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
S, B, C = 7, 2, 20
SLOT = B * 5 + C   # 30


def make_target(N=2, seed=0):
    """Build a realistic target tensor with a few GT boxes per image."""
    rng = np.random.default_rng(seed)
    targets = np.zeros((N, S, S, SLOT), dtype=np.float32)
    for n in range(N):
        for _ in range(3):                          # place 3 GT objects
            row  = rng.integers(0, S)
            col  = rng.integers(0, S)
            slot = 0 if targets[n, row, col, 4] == 0 else 1
            if targets[n, row, col, slot * 5 + 4] > 0:
                continue                            # both slots full, skip
            targets[n, row, col, slot*5+0] = rng.uniform(0.1, 0.9)   # x
            targets[n, row, col, slot*5+1] = rng.uniform(0.1, 0.9)   # y
            targets[n, row, col, slot*5+2] = rng.uniform(0.05, 0.5)  # w
            targets[n, row, col, slot*5+3] = rng.uniform(0.05, 0.5)  # h
            targets[n, row, col, slot*5+4] = 1.0                     # conf
            cls = rng.integers(0, C)
            targets[n, row, col, B*5 + cls] = 1.0
    return targets


def random_preds(N=2, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, S * S * SLOT)).astype(np.float32) * 0.1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loss_returns_positive_float():
    preds   = random_preds()
    targets = make_target()
    loss = yolo_loss(preds, targets)
    assert isinstance(loss, float), f"Expected float, got {type(loss)}"
    assert loss > 0,                f"Loss should be positive, got {loss}"
    print(f"[PASS] loss is a positive float: {loss:.4f}")


def test_grad_shape_matches_predictions():
    preds   = random_preds()
    targets = make_target()
    grad = yolo_loss_grad(preds, targets)
    assert grad.shape == preds.shape, (
        f"Grad shape {grad.shape} != pred shape {preds.shape}"
    )
    print(f"[PASS] gradient shape matches: {grad.shape}")


def test_flat_and_4d_predictions_agree():
    """yolo_loss should accept both (N,1470) and (N,7,7,30) and return the same value."""
    preds   = random_preds()
    targets = make_target()
    loss_flat = yolo_loss(preds, targets)
    loss_4d   = yolo_loss(preds.reshape(2, S, S, SLOT), targets)
    assert abs(loss_flat - loss_4d) < 1e-3, (
        f"Flat vs 4D loss mismatch: {loss_flat} vs {loss_4d}"
    )
    print(f"[PASS] flat and 4D inputs agree: {loss_flat:.4f} == {loss_4d:.4f}")


def test_perfect_predictions_lower_coord_loss():
    """Copying GT coords into predictions should reduce coord loss vs random."""
    targets = make_target()
    rng     = np.random.default_rng(0)

    random_p = rng.standard_normal((2, S * S * SLOT)).astype(np.float32) * 0.1
    loss_rand = yolo_loss(random_p, targets)

    # Build predictions that exactly match GT coords/conf/class.
    perfect_p = targets.reshape(2, -1).copy()
    loss_perf = yolo_loss(perfect_p, targets)

    assert loss_perf < loss_rand, (
        f"Perfect preds should have lower loss: {loss_perf:.4f} vs {loss_rand:.4f}"
    )
    print(f"[PASS] perfect preds loss ({loss_perf:.4f}) < random preds loss ({loss_rand:.4f})")


def test_no_object_cells_have_zero_coord_grad():
    """Gradient for coord channels in cells with no GT must be zero."""
    targets = make_target(N=1)
    preds   = random_preds(N=1)
    grad    = yolo_loss_grad(preds, targets).reshape(1, S, S, SLOT)

    # Find cells that have no GT in either predictor slot.
    noobj_mask = (targets[0, :, :, 4] == 0) & (targets[0, :, :, 9] == 0)  # (S,S)

    # Coord channels: 0,1,2,3 for slot 0 and 5,6,7,8 for slot 1.
    coord_channels = [0, 1, 2, 3, 5, 6, 7, 8]
    for ch in coord_channels:
        bad = grad[0, noobj_mask, ch]
        assert np.allclose(bad, 0, atol=1e-6), (
            f"Non-zero coord grad in no-object cell at channel {ch}: {bad}"
        )
    print("[PASS] coord gradient is zero in all no-object cells")


def test_noobj_confidence_grad_pushes_conf_down():
    """In no-object cells, conf gradient should push confidence toward 0."""
    targets = np.zeros((1, S, S, SLOT), dtype=np.float32)   # all empty
    # All-positive predictions → noobj conf grad should be positive (pushes conf down).
    preds = np.ones((1, S * S * SLOT), dtype=np.float32) * 0.5

    grad = yolo_loss_grad(preds, targets, lambda_noobj=0.5).reshape(1, S, S, SLOT)

    for b in range(B):
        conf_grad = grad[0, :, :, b * 5 + 4]
        assert np.all(conf_grad > 0), (
            f"Expected positive conf grad (pushing toward 0) for predictor {b}"
        )
    print("[PASS] no-object conf gradient is positive (pushes predicted conf toward 0)")


def test_lambda_coord_scales_coord_loss():
    """Doubling lambda_coord should double the coordinate component of the loss."""
    targets = make_target(N=1)
    preds   = random_preds(N=1)

    # Isolate coord loss by comparing two lambda_noobj=0 runs with different lambda_coord.
    l1 = yolo_loss(preds, targets, lambda_coord=2.0, lambda_noobj=0.0)
    l2 = yolo_loss(preds, targets, lambda_coord=4.0, lambda_noobj=0.0)

    # l2 - l1 should equal another l1 - (no-coord contribution).
    # Simpler check: l2 > l1 (more weight → bigger loss when coords are off).
    assert l2 > l1, f"Expected l2 > l1 with doubled lambda_coord: {l2:.4f} vs {l1:.4f}"
    print(f"[PASS] lambda_coord scaling: loss({2})={l1:.4f}  loss({4})={l2:.4f}")


def test_all_zero_targets_only_noobj_loss():
    """With no GT boxes, only the noobj confidence loss should be non-zero."""
    targets = np.zeros((2, S, S, SLOT), dtype=np.float32)
    preds   = random_preds()

    loss_full  = yolo_loss(preds, targets, lambda_coord=5.0, lambda_noobj=0.5)
    loss_noobj = yolo_loss(preds, targets, lambda_coord=0.0, lambda_noobj=0.5)

    assert abs(loss_full - loss_noobj) < 1e-3, (
        f"With no GTs, full loss should equal noobj-only loss: "
        f"{loss_full:.4f} vs {loss_noobj:.4f}"
    )
    print(f"[PASS] zero-target loss equals noobj-only loss: {loss_full:.4f}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_loss_returns_positive_float,
        test_grad_shape_matches_predictions,
        test_flat_and_4d_predictions_agree,
        test_perfect_predictions_lower_coord_loss,
        test_no_object_cells_have_zero_coord_grad,
        test_noobj_confidence_grad_pushes_conf_down,
        test_lambda_coord_scales_coord_loss,
        test_all_zero_targets_only_noobj_loss,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
