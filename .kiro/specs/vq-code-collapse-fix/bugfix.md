# Bugfix Requirements Document

## Introduction

Phase I VQ training suffers from complete codebook collapse: only 1 out of 10 codebook entries is ever selected (`used_code_count=1`). The root cause is the encoder's summary mechanism — a single-layer LSTM taking only the last hidden state `h_n`, then projecting 128→16 — which produces near-constant `z_e` vectors across all input trajectories. Since all `z_e` are nearly identical, they all quantize to the same codebook entry, and the decoder learns to ignore `z_q` entirely.

The fix requires two changes: (1) improve the encoder summary mechanism to produce diverse `z_e` representations, and (2) add a continuous latent pretraining phase where the encoder-decoder trains with continuous `z_e` (no quantization) before switching to VQ training, giving the encoder time to learn meaningful representations before quantization pressure is applied.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the encoder processes different demonstration trajectories THEN the system produces near-identical `z_e` vectors for all inputs due to the single-layer LSTM last-hidden-state bottleneck (128→16 projection), resulting in negligible variance across the batch.

1.2 WHEN VQ quantization is applied from epoch 1 with near-constant `z_e` vectors THEN the system maps all samples to a single codebook entry (`used_code_count=1`), constituting complete codebook collapse.

1.3 WHEN the decoder receives the same `z_q` for every sample in every batch THEN the system learns to ignore `z_q` and relies solely on states for reconstruction, making the latent code informationally dead.

1.4 WHEN training completes with codebook collapse THEN the system fails Phase I validation with hard failure: `used_code_count ≤ 1` or `dominant_code_ratio ≥ 0.99`.

### Expected Behavior (Correct)

2.1 WHEN the encoder processes different demonstration trajectories THEN the system SHALL produce diverse `z_e` vectors that capture trajectory-level variation, using an improved summary mechanism (e.g., temporal attention pooling over all LSTM hidden states instead of only the last hidden state).

2.2 WHEN training begins THEN the system SHALL first run a continuous latent pretraining phase (without VQ quantization) so the encoder learns to produce meaningful, diverse `z_e` representations before quantization pressure is applied, then transition to full VQ training.

2.3 WHEN VQ quantization is active after pretraining THEN the system SHALL utilize multiple codebook entries (`used_code_count > 1`), avoiding complete codebook collapse.

2.4 WHEN training completes THEN the system SHALL pass Phase I validation without hard failures related to codebook collapse (`used_code_count > 1` and `dominant_code_ratio < 0.99`).

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the encoder receives input trajectories of shape `(batch, h, state_dim)` with actions and rewards THEN the system SHALL CONTINUE TO produce `z_e` of shape `(batch, 16)` with gradients flowing through the encoder.

3.2 WHEN VQ quantization is performed THEN the system SHALL CONTINUE TO use nearest-neighbor lookup with straight-through gradient estimator, producing `z_q_st` of shape `(batch, 16)` and valid indices in range `[0, K)`.

3.3 WHEN the decoder receives states and `z_q` THEN the system SHALL CONTINUE TO produce action logits of shape `(batch, h, 3)` with valid action predictions in `{0, 1, 2}`.

3.4 WHEN the full VQ loss is computed THEN the system SHALL CONTINUE TO use the formula `L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²` during the VQ training phase.

3.5 WHEN the model is saved after training THEN the system SHALL CONTINUE TO save encoder, codebook, and decoder state dicts along with training history and config to the result directory.

3.6 WHEN Phase I validation is executed THEN the system SHALL CONTINUE TO check codebook usage, loss decomposition, reconstruction accuracy, and report hard failures and soft warnings.
