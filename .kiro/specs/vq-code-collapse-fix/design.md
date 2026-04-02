# VQ Code Collapse Fix — Bugfix Design

## Overview

Phase I VQ training suffers from complete codebook collapse because the encoder's summary mechanism (single LSTM last-hidden-state → linear projection 128→16) produces near-constant `z_e` vectors across all trajectories. This causes all samples to quantize to a single codebook entry, rendering the latent code informationally dead.

The fix has two parts:
1. Replace the last-hidden-state summary with temporal attention pooling over all LSTM hidden states, enabling the encoder to produce diverse `z_e` vectors.
2. Add a continuous latent pretraining phase at the start of training where the encoder-decoder trains without VQ quantization, giving the encoder time to learn meaningful representations before quantization pressure is applied.

## Glossary

- **Bug_Condition (C)**: The encoder summary mechanism uses only `h_n` (last hidden state) from a single-layer LSTM, producing near-constant `z_e` across different trajectories, which causes all samples to map to one codebook entry.
- **Property (P)**: The encoder should produce diverse `z_e` vectors that capture trajectory-level variation, and training should utilize multiple codebook entries (`used_code_count > 1`).
- **Preservation**: VQ loss formula, decoder architecture, codebook architecture, output shapes `z_e: (batch, 16)`, `z_q: (batch, 16)`, `logits: (batch, h, 3)`, and all paper hyperparameters (K=10, latent_dim=16, hidden_dim=128, etc.) must remain unchanged.
- **VQEncoder**: The LSTM-based encoder in `src/phase1/vq_encoder.py` that encodes demonstration trajectories `(s_demo, a_demo, r_demo)` into continuous embeddings `z_e ∈ R^16`.
- **Temporal Attention Pooling**: A learned attention mechanism that computes a weighted sum over all LSTM hidden states `H = (h_1, ..., h_T)` instead of using only `h_T`, producing a context vector that captures information from the entire sequence.
- **Continuous Latent Pretraining**: A warmup phase where the encoder-decoder trains with continuous `z_e` (bypassing VQ quantization), using loss `L = L_rec` only, before transitioning to full VQ training.

## Bug Details

### Bug Condition

The bug manifests when the encoder processes batches of diverse demonstration trajectories but produces near-identical `z_e` vectors for all of them. The `VQEncoder.forward()` takes only `h_n` (the final hidden state of a single-layer LSTM), which is a severe information bottleneck — the LSTM must compress all 72 timesteps of trajectory information into a single 128-dim vector, and the subsequent 128→16 linear projection further collapses variance. Combined with immediate VQ quantization pressure from epoch 1, the encoder never learns to differentiate trajectories.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type {encoder: VQEncoder, trajectories: Tensor[batch, h, features]}
  OUTPUT: boolean

  z_e_batch := encoder.forward(trajectories)          -- (batch, 16)
  z_e_variance := variance(z_e_batch, dim=0)           -- (16,)
  mean_variance := mean(z_e_variance)

  RETURN mean_variance < EPSILON                        -- near-zero variance across batch
         AND codebook_used_count(z_e_batch) <= 1        -- all map to same entry
END FUNCTION
```

### Examples

- **Batch of 256 diverse trajectories**: Expected: `z_e` vectors spread across latent space, mapping to multiple codebook entries. Actual: all `z_e` vectors nearly identical (variance < 1e-6), all map to codebook entry 0, `used_code_count=1`.
- **Two maximally different trajectories** (all-short vs all-long): Expected: distinct `z_e` vectors mapping to different codebook entries. Actual: `z_e` vectors differ by < 1e-4 in L2 norm, both map to same entry.
- **Training after 100 epochs**: Expected: `codebook_perplexity ≈ K`, `dominant_code_ratio < 0.5`. Actual: `codebook_perplexity ≈ 1.0`, `dominant_code_ratio ≈ 1.0`, Phase I validation hard failure.
- **Edge case — single-timestep trajectory (h=1)**: Even with attention pooling, a single timestep should still produce valid `z_e` of shape `(batch, 16)`.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- `z_e` output shape must remain `(batch, 16)` with gradients flowing through the encoder
- VQ quantization must continue to use nearest-neighbor lookup with straight-through gradient estimator, producing `z_q_st` of shape `(batch, 16)` and valid indices in `[0, K)`
- Decoder must continue to produce action logits of shape `(batch, h, 3)` with valid actions in `{0, 1, 2}`
- VQ loss formula `L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²` must remain unchanged during the VQ training phase
- Model saving must continue to save encoder, codebook, and decoder state dicts with training history and config
- Phase I validation must continue to check codebook usage, loss decomposition, reconstruction accuracy
- All paper hyperparameters unchanged: K=10, latent_dim=16, hidden_dim=128, state_dim=45, action_dim=3, vq_beta0=0.25, horizon=72
- Codebook architecture (`VQCodebook`) unchanged
- Decoder architecture (`VQDecoder`) unchanged

**Scope:**
All inputs that do NOT involve the encoder summary mechanism or the training phase logic should be completely unaffected by this fix. This includes:
- Codebook quantization logic (nearest-neighbor, straight-through, commitment loss)
- Decoder forward pass (MLP with state + z_q concatenation)
- DP trajectory generation and caching
- Phase I validation logic
- Data loading and preprocessing

## Hypothesized Root Cause

Based on the bug description, the most likely issues are:

1. **Last-Hidden-State Information Bottleneck**: The single-layer LSTM's `h_n` must compress all 72 timesteps into one 128-dim vector. For long sequences, `h_n` is dominated by the most recent timesteps and loses early trajectory information. The subsequent 128→16 projection further destroys variance, producing near-constant outputs regardless of input trajectory diversity.

2. **Immediate VQ Quantization Pressure**: From epoch 1, the commitment loss `β₀ × ||z_e - sg[z_q]||²` pushes all `z_e` toward the nearest codebook entry. Since the encoder hasn't yet learned to differentiate trajectories, all `z_e` start near the same point and get pulled toward the same codebook vector, creating a self-reinforcing collapse loop.

3. **Compounding Effect**: These two causes compound — the weak encoder produces low-variance `z_e`, VQ pressure collapses them further, the decoder learns to ignore `z_q`, and the encoder receives no gradient signal to differentiate trajectories.

## Correctness Properties

Property 1: Bug Condition — Encoder Produces Diverse z_e Vectors

_For any_ batch of distinct demonstration trajectories processed by the fixed encoder (with temporal attention pooling), the output `z_e` vectors SHALL have non-negligible variance across the batch dimension, and when quantized, SHALL map to more than one codebook entry.

**Validates: Requirements 2.1, 2.3**

Property 2: Preservation — Output Shape and Gradient Flow Invariants

_For any_ input trajectories `(s_demo, a_demo, r_demo)` with arbitrary batch size and sequence length, the fixed encoder SHALL produce `z_e` of shape `(batch, 16)` with gradients flowing through all encoder parameters, and the full pipeline (encoder → codebook → decoder) SHALL produce action logits of shape `(batch, h, 3)` with the VQ loss formula unchanged during the VQ phase.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**


## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `src/phase1/vq_encoder.py`

**Class**: `VQEncoder`

**Specific Changes**:
1. **Add Temporal Attention Pooling Layer**: Add a learned attention mechanism that computes a weighted sum over all LSTM hidden states instead of using only `h_n`.
   - Add `self.attn_score = nn.Linear(hidden_dim, 1)` to compute per-timestep attention scores
   - In `forward()`, run LSTM to get full output `H: (batch, h, hidden_dim)` instead of just `h_n`
   - Compute attention weights: `alpha = softmax(attn_score(H), dim=1)` → `(batch, h, 1)`
   - Compute context vector: `context = sum(alpha * H, dim=1)` → `(batch, hidden_dim)`
   - Project: `z_e = self.projection(context)` → `(batch, latent_dim)`

2. **Remove Last-Hidden-State Usage**: Replace `h_n.squeeze(0)` with the attention-pooled context vector. The `self.projection` layer remains unchanged (still `hidden_dim → latent_dim`).

---

**File**: `src/config.py`

**Class**: `Config`

**Specific Changes**:
3. **Add Pretrain Epochs Parameter**: Add `pretrain_epochs: int = 10` field to `Config` dataclass — the number of epochs for continuous latent pretraining before VQ training begins.

4. **Add CLI Argument**: Add `--pretrain-epochs` argument to `parse_args()` so the pretrain epoch count can be overridden from the command line.

---

**File**: `scripts/train_phase1.py`

**Function**: `main()`

**Specific Changes**:
5. **Two-Phase Training Loop**: Split the training loop into two phases:
   - **Phase A (Continuous Pretrain)**: For `pretrain_epochs` epochs, bypass VQ quantization. Pass `z_e` directly to the decoder (instead of `z_q_st`). Loss = `L_rec` only (no commitment loss, no encoder commitment). This lets the encoder learn to produce diverse, meaningful `z_e` before quantization pressure.
   - **Phase B (VQ Train)**: For the remaining `phase1_epochs` epochs, use the full VQ pipeline as before: encode → quantize → decode, with full loss `L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²`.

6. **Update Paper Settings Guard**: Update `PAPER_PHASE1_SPEC` and `assert_paper_phase1_settings()` to include `pretrain_epochs` so the guard accepts the new parameter.

7. **Update Model Checkpoint**: Include `pretrain_epochs` in the saved config dict.

---

**File**: `tests/test_vq.py`

**Specific Changes**:
8. **Update Encoder Init Tests**: Update `TestVQEncoderInit` to verify the new `attn_score` layer exists and has correct dimensions (`hidden_dim → 1`).

9. **Add Attention Pooling Tests**: Add tests verifying that the attention mechanism produces valid weights (sum to 1, non-negative) and that different-length sequences produce valid outputs.

10. **Preserve Existing Tests**: All existing codebook, decoder, and property-based tests must continue to pass without modification.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Create a test that instantiates the current (unfixed) `VQEncoder`, feeds it a batch of diverse trajectories, and measures `z_e` variance. Run on unfixed code to observe near-zero variance confirming the last-hidden-state bottleneck.

**Test Cases**:
1. **z_e Variance Test**: Feed 32 random trajectories through the encoder, measure per-dimension variance of `z_e`. (will fail on unfixed code — variance near zero)
2. **Codebook Usage Test**: Feed batch through encoder → codebook, count unique indices. (will fail on unfixed code — `used_code_count=1`)
3. **Diverse Input Sensitivity Test**: Feed two maximally different trajectories, measure L2 distance of their `z_e`. (will fail on unfixed code — distance near zero)

**Expected Counterexamples**:
- `z_e` variance across a batch of 32 diverse trajectories is < 1e-4
- All 32 samples map to the same codebook index
- Possible causes: last-hidden-state bottleneck compresses all trajectory info into near-constant output

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  z_e := fixed_encoder(input.trajectories)
  variance := var(z_e, dim=0).mean()
  indices := codebook.quantize(z_e).indices
  ASSERT variance > EPSILON
  ASSERT unique_count(indices) > 1
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  -- Shape invariants
  ASSERT fixed_encoder(input).shape == (batch, 16)
  -- Codebook behavior unchanged
  ASSERT codebook.quantize(z_e).z_q_st.shape == (batch, 16)
  ASSERT codebook.quantize(z_e).indices IN [0, K)
  -- Decoder behavior unchanged
  ASSERT decoder(states, z_q).shape == (batch, h, 3)
  -- VQ loss formula unchanged
  ASSERT total_loss == L_rec + commitment_loss + beta0 * encoder_commitment
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain (varying batch sizes, sequence lengths)
- It catches edge cases that manual unit tests might miss (e.g., batch_size=1, seq_len=1)
- It provides strong guarantees that shape invariants and loss formulas are unchanged

**Test Plan**: Observe behavior on UNFIXED code first for shape outputs, gradient flow, and loss decomposition, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Shape Preservation**: Verify `z_e` shape is `(batch, 16)` for arbitrary batch sizes and sequence lengths after fix
2. **Gradient Flow Preservation**: Verify gradients flow through all encoder parameters (LSTM + attention + projection) after fix
3. **VQ Loss Formula Preservation**: Verify loss decomposition `L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²` is unchanged during VQ phase
4. **Decoder Output Preservation**: Verify decoder produces `(batch, h, 3)` logits with valid actions

### Unit Tests

- Test `VQEncoder.__init__` includes `attn_score` layer with correct dimensions
- Test attention weights sum to 1 and are non-negative for various sequence lengths
- Test `z_e` output shape `(batch, 16)` for edge cases (batch=1, seq_len=1)
- Test gradient flows through attention layer, LSTM, and projection
- Test that `Config.pretrain_epochs` defaults to 10 and is overridable via CLI

### Property-Based Tests

- Generate random batch sizes and sequence lengths, verify `z_e` shape is always `(batch, 16)` (updated Property 12)
- Generate random `z_e` vectors, verify codebook quantization still selects true nearest neighbor (Property 13 unchanged)
- Generate random states and `z_q`, verify decoder outputs valid actions (Property 14 unchanged)
- Generate random `z_e`, verify VQ loss decomposition is correct (Property 15 unchanged)

### Integration Tests

- Test full pipeline: encoder (with attention) → codebook → decoder produces correct output shapes
- Test continuous pretrain phase: verify `z_e` is passed directly to decoder (no quantization) and loss = `L_rec` only
- Test VQ phase: verify full loss formula is used after pretrain epochs complete
- Test phase transition: verify training switches cleanly from continuous pretrain to VQ training
