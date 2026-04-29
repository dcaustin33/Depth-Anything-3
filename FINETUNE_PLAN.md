# Minimal Fine-Tuning Plan: DA3METRIC-LARGE on 256×256 Metric Depth Data

This is a plan — not a working script — for fine-tuning the pretrained
`DA3METRIC-LARGE` checkpoint on a dataset of 256×256 RGB images with
ground-truth metric depth. It consolidates every assumption, unit convention,
and code touch-point we identified when stepping through the repo.

## 0. Assumptions (resolve these before writing code)

1. **Images:** 256×256 RGB. Preprocessor will resize to 504×504
   (`upper_bound_resize`, longest side → 504, already divisible by 14).
2. **GT depth:** per-pixel metric depth in meters, rendered/measured as if
   the camera had **focal length = 300 px at 256×256**. Every image in the
   training set has a known focal in this form.
3. **Target:** fine-tune the depth head (and optionally sky head) on this
   data while preserving the pretrained convention so the existing
   `scripts/run_metric_depth.py` inference formula keeps working for users
   with any focal.
4. **Hardware:** single GPU, small-to-medium batch. No DDP in v0.

> ⚠️ **Before writing any code**, run a sanity check: take one 256×256 image
> with its focal=300-at-256 GT, run `scripts/run_metric_depth.py` on it with
> `--focal-px 300`, and compare the predicted meters to your GT meters
> pixel-wise. If the mean ratio is ~1×, your convention matches. If it's
> off by a consistent factor (0.5×, 2×), that factor reveals what your GT
> convention really is. Do not proceed until this matches.

## 1. Unit conventions (the part that is easy to get wrong)

- Preprocess resizes 256 → 504. Therefore: `focal_px_processed = 300 × (504/256) = 590.625 px`.
- Pretrained model outputs `raw_depth` such that `depth_m = focal_px_processed × raw / 300`.
- Your GT in meters converts to "raw canonical units" by inverting:
  `gt_canonical = gt_m × 300 / focal_px_processed = gt_m × (300 / 590.625) ≈ gt_m × 0.5079`.
- This is the quantity the network's `raw_depth` output should equal.
- Equivalently, since the depth head is `raw = exp(logit)`, the quantity
  `log(gt_canonical)` is what the pre-exp logit should equal.

Keep the conversion factor in one constant (e.g. `CANONICAL_OVER_PROCESSED = 300 / 590.625`) and do not sprinkle it across files.

## 2. Architecture touch-points

All activation and unit behavior lives here:

- `src/depth_anything_3/model/dpt.py:48` — depth head `activation="exp"` (kept).
- `src/depth_anything_3/model/dpt.py:244-256` — `main_logits` → `exp(main_logits)` → stored under `head_main` (default `"depth"`).
- `src/depth_anything_3/model/dpt.py:286-309` — activation dispatch.
- `src/depth_anything_3/model/da3.py:140` — `_process_depth_head` calls `self.head(...)`.
- `src/depth_anything_3/model/da3.py:155-179` — `_process_mono_sky_estimation` rewrites sky pixels to a quantile-derived max depth **at inference**; this must be **bypassed during training** so gradients reach sky pixels naturally (or so we can mask them out cleanly in the loss).
- `scripts/run_metric_depth.py:11,27,63,69` — the inference-side scaling. Not changed by training, but the contract we must preserve.

No architectural change is required to start. If we want log-space loss on
pre-exp logits instead of post-exp values, we have two options:

1. Override the head's `activation` to `"linear"` during training and apply
   `exp` manually only when logging "human-readable" depth. Cleanest.
2. Keep `"exp"` and compute `loss = F.l1_loss(torch.log(pred), torch.log(gt_canonical))`. Numerically equivalent to (1) for L1/L2, and requires no code change.

Start with option 2 — zero code changes to the model.

## 3. Data pipeline

### 3.1 Inputs the training loop must produce per sample

- `rgb_504`: `(3, 504, 504)` float tensor, ImageNet-normalized, exactly matching `src/depth_anything_3/utils/io/input_processor.py` so training distribution matches inference distribution.
- `gt_504_m`: `(504, 504)` float tensor, GT depth in meters, bilinearly upsampled from 256 to 504 with the same aspect-ratio-preserving resize the RGB uses.
- `valid`: `(504, 504)` bool tensor. `True` where GT is finite, positive, and not flagged as sky/invalid. Must exclude `gt<=0`, `gt==inf`, and (if you have labels) explicit sky pixels.
- `focal_px_input`: scalar. For this dataset, always `300.0` (focal at 256 resolution).

### 3.2 GT conversion to canonical units

Per sample:

```python
PROC_RES = 504
ORIG_RES = 256
CANONICAL_FOCAL = 300.0

focal_px_processed = focal_px_input * (PROC_RES / ORIG_RES)   # 590.625 when focal=300
gt_canonical = gt_504_m * (CANONICAL_FOCAL / focal_px_processed)
```

If every image has focal=300, the scalar is constant ≈ 0.5079 — precompute it.

### 3.3 Reuse the existing preprocessor

Prefer calling `depth_anything_3.utils.io.input_processor.InputProcessor` on
RGB rather than reimplementing resize + normalize. Mirror its resize
parameters for GT depth — same `process_res=504`, same `upper_bound_resize`,
but use `cv2.INTER_LINEAR` (or nearest, benchmark both) instead of CUBIC/AREA.

> **Do not use CenterCrop on GT unless RGB was also cropped.** Keep the
> spatial alignment identical or the loss is meaningless.

## 4. Model loading and parameter freezing

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
model.train()
```

`DepthAnything3` wraps `DepthAnything3Net` (see `src/depth_anything_3/model/da3.py:40`). Freezing recommendation (v0):

- **Freeze:** `backbone` (DINOv2). These features are general; unfreezing risks
  catastrophic forgetting and needs more data + tiny LR.
- **Train:** `head` only. That's where unit conventions and scale live.
- **Freeze or drop:** `cam_dec`, `cam_enc`, `gs_head`, `gs_adapter`. Not used
  for monocular metric depth; leaving them unfrozen but unused will let them
  drift. Explicitly `requires_grad_(False)`.

```python
for p in model.parameters():
    p.requires_grad_(False)
for p in model.head.parameters():
    p.requires_grad_(True)
```

Confirm by printing a count of trainable params — should be roughly the DPT
head size, not the full 0.35B.

### 4.1 Bypass the sky-max rewrite during training

`_process_mono_sky_estimation` (`da3.py:155-179`) overwrites depth in sky
regions at a scene-wide quantile. That's an inference convenience; during
training it destroys gradient flow. Options:

1. Call `model.net.head(feats, H, W, patch_start_idx=0)` directly, skipping
   `_process_mono_sky_estimation`.
2. Monkey-patch `_process_mono_sky_estimation` to return `output` unchanged
   when `model.training` is True.
3. Subclass `DepthAnything3Net` with a training-mode forward that only runs
   the depth head.

(2) is the minimum-intrusion choice. (3) is the cleanest long-term.

## 5. Loss

Start with scale-aware log-L1 over valid pixels:

```python
pred_raw = output.depth                     # (B, 1, 504, 504) or (B, 504, 504), = exp(logit)
gt_canonical_batch = ...                    # same shape
valid_batch = ...

eps = 1e-6
loss = F.l1_loss(
    torch.log(pred_raw.clamp_min(eps))[valid_batch],
    torch.log(gt_canonical_batch.clamp_min(eps))[valid_batch],
)
```

Optional additions, each behind a flag:

- **SiLog term** for scale robustness if the metric-calibration quality of
  the GT varies across samples.
- **Gradient-matching term** on `∇log(pred) − ∇log(gt)` to sharpen edges.
- **Confidence NLL** if you want the `depth_conf` head to learn aleatoric
  uncertainty; this only kicks in if the head's `output_dim > 1` (check the
  loaded config).

Do not do MSE on raw meters — gradients for distant pixels blow up.

## 6. Training loop (pseudocode)

```python
device = torch.device("cuda")
model = load_and_freeze()
model.to(device)

opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.0,
)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    for rgb_504, gt_504_m, valid, focal_px_input in loader:
        rgb_504 = rgb_504.to(device)
        gt_504_m = gt_504_m.to(device)
        valid = valid.to(device)

        focal_processed = focal_px_input * (504 / 256)
        gt_canonical = gt_504_m * (300.0 / focal_processed)[:, None, None]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Shape: (B, N=1, 3, 504, 504) — DA3 expects a view dim.
            x = rgb_504.unsqueeze(1)
            out = model.net(x)  # skip the app-level inference wrappers
            pred_raw = out.depth.squeeze(1).squeeze(-1)  # (B, 504, 504) — verify shapes

        eps = 1e-6
        loss = F.l1_loss(
            torch.log(pred_raw.clamp_min(eps))[valid],
            torch.log(gt_canonical.clamp_min(eps))[valid],
        )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
```

Key points that will bite you:

- **DA3 expects a view/batch structure** `(B, N, 3, H, W)` not `(B, 3, H, W)`.
  Always unsqueeze a view dim of 1 for monocular.
- **`out.depth` ordering**. Inference-time output processor squeezes to
  `(N, H, W)` (`src/depth_anything_3/utils/io/output_processor.py:87`).
  During training you'll pull the raw tensor from the head directly —
  shape is `(B, S, 1, H, W)`. Verify with a dry-run print before writing loss.
- **DO NOT call `model.inference(...)`** in the training loop; that runs
  preprocessing, pose estimation, sky post-processing, and converts to numpy.
  Call the underlying `nn.Module` (`model.net` or similar — check the API).

## 7. Validation and logging

Every N steps, on a held-out batch:

1. Forward pass, same as training.
2. Convert back to meters for human-readable metrics:
   `pred_m = pred_raw * focal_processed / 300`.
3. Metrics to log:
   - Mean abs log error in canonical units.
   - AbsRel: `mean(|pred_m - gt_m| / gt_m)` over valid pixels.
   - δ₁ accuracy: `mean(max(pred_m/gt_m, gt_m/pred_m) < 1.25)`.
4. Visualize one RGB / GT / pred triplet to TensorBoard or a saved PNG.
5. **Inference parity check**: once per epoch, save the checkpoint, then run
   `scripts/run_metric_depth.py` on a fixed validation image with the real
   focal. The reported meters should be sane. If they blow up, the
   canonical-unit convention broke somewhere.

## 8. Risks and how to catch them

| Risk | Signal | Mitigation |
|---|---|---|
| GT convention mismatch (focal at 256 vs 504) | Loss plateaus ~log(2) above zero; metrics show consistent 2× or 0.5× scale error | Run the Sec. 0 sanity check; verify by training-set mean ratio of pred/gt before any gradient step |
| Forgetting the view dimension | Backbone crashes or outputs garbage shape | Always `x = rgb.unsqueeze(1)` |
| Sky pixels dominating loss | Loss drops but predictions become smooth/featureless | Mask sky in `valid`; optionally add small sky-head BCE loss using same labels |
| DPT head drift because confidence channel is unused but unfrozen | Silent slow degradation | Either train confidence with an NLL term or freeze that output channel |
| `_process_mono_sky_estimation` clobbering pred in training | `pred_raw` shows huge plateaus at one value per image | Skip that step in training (Sec. 4.1) |
| Backbone accidentally trainable | Loss decreases fast but inference regresses on out-of-distribution images | Assert `sum(p.numel() for p in model.backbone.parameters() if p.requires_grad) == 0` at start of every epoch |

## 9. v0 deliverable checklist

- [ ] Sanity check from Sec. 0 passes.
- [ ] Dataset class produces `(rgb_504, gt_504_m, valid, focal_px_input)` tuples.
- [ ] Preprocessor path reused for RGB; custom but mirrored path for GT.
- [ ] Model loaded, backbone + unused heads frozen, trainable-param count logged.
- [ ] Sky-max post-processing bypassed in training mode.
- [ ] Forward/backward works on a single batch without shape errors.
- [ ] Loss in log space over valid pixels, stable for 100 steps.
- [ ] Validation AbsRel + δ₁ logged; reasonable values (AbsRel < 0.3, δ₁ > 0.5) after 1 epoch on even a small dataset.
- [ ] Saved checkpoint reloadable by `DepthAnything3.from_pretrained(...)` path OR by a custom loader that mirrors `depth_anything_3.utils.model_loading.load_pretrained_weights`.
- [ ] `scripts/run_metric_depth.py` on a held-out image still produces sensible meters using the fine-tuned checkpoint.

## 10. Future extensions (not v0)

- Unfreeze DINOv2 at 1e-6 after 1–2 epochs of head-only training.
- Per-sample variable focal with the canonical-unit conversion per-example,
  once the single-focal case is stable.
- Sky head training with a BCE loss against ground-truth sky masks.
- Confidence head training with Laplacian NLL.
- Multi-resolution training — random `process_res ∈ {364, 504, 644}` to
  improve robustness.
- DDP once the loop is validated.
