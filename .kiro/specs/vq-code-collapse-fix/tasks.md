# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** — Encoder Produces Diverse z_e Vectors
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior — it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the encoder produces near-constant z_e vectors
  - **Scoped PBT Approach**: Use Hypothesis to generate batches of random trajectories (varying batch_size in [4,32], seq_len in [1,72]) and assert that the encoder produces z_e with non-negligible variance
  - Test file: `tests/test_vq.py` — add class `TestPropBugConditionDiverseZe`
  - Instantiate the current (unfixed) `VQEncoder(state_dim=45)` with random weights
  - Feed batch of diverse random trajectories `(s_demo, a_demo, r_demo)` through encoder
  - Compute `z_e_variance = var(z_e, dim=0).mean()` across the batch dimension
  - Assert `z_e_variance > 1e-4` (from Bug Condition: `mean_variance < EPSILON` indicates bug)
  - Quantize z_e through `VQCodebook(num_codes=10, code_dim=16)` and assert `unique_count(indices) > 1`
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (variance is near-zero, all indices identical — confirms the last-hidden-state bottleneck bug)
  - Document counterexamples found: e.g., "batch of 32 trajectories all produce z_e with variance < 1e-6, all map to codebook entry 0"
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 2.1, 2.3_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** — Output Shape and Gradient Flow Invariants
  - **IMPORTANT**: Follow observation-first methodology
  - Test file: `tests/test_vq.py` — add class `TestPropPreservationShapeGradient`
  - Observe on UNFIXED code: `VQEncoder(state_dim=45).forward(s, a, r)` produces shape `(batch, 16)` for arbitrary batch/seq_len
  - Observe on UNFIXED code: gradients flow through all encoder parameters (LSTM + projection) after backward
  - Observe on UNFIXED code: full pipeline encoder → codebook → decoder produces `(batch, h, 3)` logits
  - Observe on UNFIXED code: VQ loss formula `L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²` holds
  - Write property-based tests using Hypothesis with `batch_size=st.integers(1, 32)` and `seq_len=st.integers(1, 72)`:
    - Assert `z_e.shape == (batch_size, 16)` for all generated inputs
    - Assert gradients exist on all encoder parameters after backward pass
    - Assert `decoder(states, z_q).shape == (batch_size, seq_len, 3)`
    - Assert VQ loss decomposition: `total_loss ≈ rec_loss + commitment_loss + β₀ × encoder_commitment`
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (shape invariants and loss formula hold on current code)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3. Fix VQ code collapse — temporal attention pooling + continuous pretrain

  - [x] 3.1 Replace last-hidden-state with temporal attention pooling in VQEncoder
    - File: `src/phase1/vq_encoder.py`
    - Add `self.attn_score = nn.Linear(hidden_dim, 1)` in `__init__`
    - In `forward()`: use LSTM full output `H: (batch, h, hidden_dim)` instead of `h_n`
    - Compute attention weights: `alpha = softmax(attn_score(H), dim=1)` → `(batch, h, 1)`
    - Compute context vector: `context = sum(alpha * H, dim=1)` → `(batch, hidden_dim)`
    - Project: `z_e = self.projection(context)` → `(batch, latent_dim)`
    - Remove `h_n.squeeze(0)` usage
    - Update `TestVQEncoderInit` to verify `attn_score` layer exists with shape `(hidden_dim, 1)`
    - Add unit tests for attention weights (sum to 1, non-negative) and edge cases (seq_len=1)
    - _Bug_Condition: isBugCondition(input) where encoder uses only h_n producing near-constant z_e_
    - _Expected_Behavior: encoder uses temporal attention pooling over all LSTM hidden states to produce diverse z_e_
    - _Preservation: z_e shape (batch, 16), gradient flow through LSTM + attn_score + projection_
    - _Requirements: 2.1, 2.3, 3.1_

  - [x] 3.2 Add pretrain_epochs config and CLI argument
    - File: `src/config.py`
    - Add `pretrain_epochs: int = 10` field to `Config` dataclass
    - Add `--pretrain-epochs` argument to `parse_args()` with `type=int, default=None`
    - Wire through `Config.from_args()` (already handled by generic field iteration)
    - _Requirements: 2.2_

  - [x] 3.3 Split training into Phase A (continuous pretrain) and Phase B (VQ training)
    - File: `scripts/train_phase1.py`
    - Phase A (epochs 1..pretrain_epochs): bypass VQ quantization, pass z_e directly to decoder, loss = L_rec only
    - Phase B (epochs pretrain_epochs+1..phase1_epochs): full VQ pipeline as before with L = L_rec + commitment + β₀ × encoder_commitment
    - Update `PAPER_PHASE1_SPEC` to include `"pretrain_epochs": 10`
    - Update `assert_paper_phase1_settings()` to check `pretrain_epochs`
    - Update model checkpoint saved config dict to include `pretrain_epochs`
    - Log phase transitions clearly
    - _Bug_Condition: immediate VQ quantization from epoch 1 compounds the collapse_
    - _Expected_Behavior: continuous pretrain lets encoder learn diverse z_e before quantization pressure_
    - _Preservation: VQ loss formula unchanged during Phase B, model save format extended_
    - _Requirements: 2.2, 2.4, 3.4, 3.5_

  - [x] 3.4 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** — Encoder Produces Diverse z_e Vectors
    - **IMPORTANT**: Re-run the SAME test from task 1 — do NOT write a new test
    - The test from task 1 encodes the expected behavior (diverse z_e, multiple codebook entries)
    - When this test passes, it confirms the temporal attention pooling fix resolves the bottleneck
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.3_

  - [x] 3.5 Verify preservation tests still pass
    - **Property 2: Preservation** — Output Shape and Gradient Flow Invariants
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Also verify existing Properties 12-15 in `tests/test_vq.py` still pass
    - Confirm all tests still pass after fix (no regressions)

- [x] 4. Checkpoint — Ensure all tests pass
  - Run full test suite: `pytest tests/test_vq.py -v`
  - Verify Properties 1-2 (new bug condition + preservation) pass
  - Verify Properties 12-15 (existing VQ dimension, nearest-neighbor, decoder actions, loss correctness) pass
  - Verify all unit tests for encoder init, attention pooling, config, and training phases pass
  - Ensure all tests pass, ask the user if questions arise
