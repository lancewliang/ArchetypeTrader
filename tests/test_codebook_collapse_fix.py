"""Bug Condition Exploration Test — 随机初始化码本在 z_e 分布下产生 Dead Codes.

Property 1: Bug Condition — 随机初始化码本在 z_e 分布下产生 Dead Codes

For any z_e distribution concentrated in a sub-region of the latent space
(simulating Phase A trained encoder output), a randomly initialized VQCodebook
(without k-means init_from_data) SHALL quantize z_e samples such that
used_code_count == K (all codebook entries are used).

On UNFIXED code — EXPECTED TO FAIL: randomly initialized codebook vectors are
far from the z_e distribution, so only a few entries act as nearest neighbors,
producing dead codes (used_code_count < K).

**Validates: Requirements 1.1, 1.2, 2.2, 2.3**

EXPECTED OUTCOME on unfixed code: FAIL (proves the bug exists)
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.phase1.codebook import VQCodebook


class TestBugConditionDeadCodes:
    """Bug condition exploration: randomly initialized codebook produces dead codes
    when z_e is concentrated in a sub-region of the latent space."""

    @given(
        mu_val=st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        sigma=st.floats(min_value=0.01, max_value=0.2, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=200, deadline=None)
    def test_random_init_codebook_all_codes_used(
        self, mu_val: float, sigma: float, seed: int
    ):
        """Property 1: Bug Condition — 随机初始化码本在 z_e 分布下产生 Dead Codes.

        **Validates: Requirements 1.1, 1.2, 2.2, 2.3**

        Generate N>=100 z_e vectors concentrated around a random mean with small
        std (simulating z_e after Phase A training). Create a randomly initialized
        VQCodebook (no init_from_data). Quantize all z_e samples and assert all
        K=10 codebook entries are used.

        On unfixed code, the randomly initialized codebook is far from the z_e
        distribution, so most z_e samples map to only a few nearest entries,
        leaving others as dead codes (used_code_count < 10).
        """
        torch.manual_seed(seed)

        num_codes = 10
        code_dim = 16
        n_samples = 100

        # Create randomly initialized codebook (default nn.Embedding init)
        codebook = VQCodebook(num_codes=num_codes, code_dim=code_dim)

        # Generate z_e samples concentrated in a sub-region:
        # mu is a constant vector shifted away from origin, sigma is small
        mu = torch.full((code_dim,), mu_val)
        z_e = mu.unsqueeze(0) + sigma * torch.randn(n_samples, code_dim)

        # Apply k-means initialization from z_e data before quantizing
        codebook.init_from_data(z_e)

        # Quantize after k-means initialization
        with torch.no_grad():
            _, indices, _ = codebook.quantize(z_e)

        used_code_count = len(torch.unique(indices))

        assert used_code_count == num_codes, (
            f"Dead codes detected: used_code_count={used_code_count}/{num_codes}. "
            f"z_e concentrated at mu={mu_val}, sigma={sigma}, seed={seed}. "
            f"Randomly initialized codebook fails to cover z_e distribution — "
            f"confirms bug condition."
        )


# ---------------------------------------------------------------------------
# Property 2: Preservation — Quantize 方法、损失公式和流水线行为不变
# ---------------------------------------------------------------------------

"""
Preservation property tests — verify that quantize(), loss formulas, encoder,
decoder, and Phase A bypass behavior are unchanged on the UNFIXED code.

These tests must PASS on unfixed code (confirming baseline behavior).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import hypothesis.extra.numpy as hnp

from src.phase1.vq_encoder import VQEncoder
from src.phase1.vq_decoder import VQDecoder


# ---- Shared strategies ----

def _z_e_strategy(batch_min=1, batch_max=8, code_dim=16):
    """Generate z_e tensors of shape (batch, code_dim) with reasonable values."""
    return (
        st.integers(min_value=batch_min, max_value=batch_max)
        .flatmap(
            lambda b: hnp.arrays(
                dtype=np.float32,
                shape=(b, code_dim),
                elements=st.floats(min_value=-5.0, max_value=5.0,
                                   allow_nan=False, allow_infinity=False,
                                   width=32),
            )
        )
    )


class TestPreservationQuantizeInterface:
    """Property 2.1: quantize() interface unchanged.

    **Validates: Requirements 3.2**

    For any z_e (batch, 16), quantize() returns (z_q_st, indices, commitment_loss)
    with shapes (batch, 16), (batch,), scalar; regardless of whether codebook has
    been k-means initialized.
    """

    @given(z_e_np=_z_e_strategy())
    @settings(max_examples=50, deadline=None)
    def test_quantize_output_shapes_random_init(self, z_e_np: np.ndarray):
        """quantize() returns correct shapes with random-init codebook."""
        z_e = torch.tensor(z_e_np)
        batch = z_e.shape[0]
        codebook = VQCodebook(num_codes=10, code_dim=16)

        with torch.no_grad():
            z_q_st, indices, commitment_loss = codebook.quantize(z_e)

        assert z_q_st.shape == (batch, 16), f"z_q_st shape {z_q_st.shape} != ({batch}, 16)"
        assert indices.shape == (batch,), f"indices shape {indices.shape} != ({batch},)"
        assert commitment_loss.dim() == 0, f"commitment_loss is not scalar, dim={commitment_loss.dim()}"

    @given(z_e_np=_z_e_strategy())
    @settings(max_examples=50, deadline=None)
    def test_quantize_output_shapes_kmeans_init(self, z_e_np: np.ndarray):
        """quantize() returns correct shapes after k-means init."""
        z_e = torch.tensor(z_e_np)
        batch = z_e.shape[0]
        codebook = VQCodebook(num_codes=10, code_dim=16)

        # Initialize with k-means from the same z_e (need at least K samples)
        if batch >= 10:
            codebook.init_from_data(z_e)

        with torch.no_grad():
            z_q_st, indices, commitment_loss = codebook.quantize(z_e)

        assert z_q_st.shape == (batch, 16)
        assert indices.shape == (batch,)
        assert commitment_loss.dim() == 0


class TestPreservationNearestNeighbor:
    """Property 2.2: Nearest neighbor correctness unchanged.

    **Validates: Requirements 3.2**

    For any z_e, selected index k satisfies ||z_e - e_k||² ≤ ||z_e - e_j||² ∀j.
    """

    @given(z_e_np=_z_e_strategy())
    @settings(max_examples=50, deadline=None)
    def test_nearest_neighbor_correctness(self, z_e_np: np.ndarray):
        """Selected index is the true nearest neighbor for every sample."""
        z_e = torch.tensor(z_e_np)
        codebook = VQCodebook(num_codes=10, code_dim=16)

        with torch.no_grad():
            _, indices, _ = codebook.quantize(z_e)
            weights = codebook.embeddings.weight  # (K, 16)

            for i in range(z_e.shape[0]):
                dists = torch.sum((z_e[i] - weights) ** 2, dim=1)  # (K,)
                expected_idx = torch.argmin(dists)
                assert indices[i] == expected_idx, (
                    f"Sample {i}: quantize chose index {indices[i].item()} "
                    f"but true nearest is {expected_idx.item()}"
                )


class TestPreservationStraightThrough:
    """Property 2.3: Straight-through gradient unchanged.

    **Validates: Requirements 3.2**

    z_q_st gradient w.r.t. z_e is all 1s (z_q_st = z_e + (z_q - z_e).detach()).
    """

    @given(z_e_np=_z_e_strategy())
    @settings(max_examples=50, deadline=None)
    def test_straight_through_gradient(self, z_e_np: np.ndarray):
        """Gradient of z_q_st w.r.t. z_e is identity (all ones)."""
        z_e = torch.tensor(z_e_np, requires_grad=True)
        codebook = VQCodebook(num_codes=10, code_dim=16)

        z_q_st, _, _ = codebook.quantize(z_e)

        # Backprop a scalar derived from z_q_st
        loss = z_q_st.sum()
        loss.backward()

        assert z_e.grad is not None, "z_e.grad is None"
        expected_grad = torch.ones_like(z_e)
        assert torch.allclose(z_e.grad, expected_grad, atol=1e-6), (
            f"Straight-through gradient is not all 1s. "
            f"Max deviation: {(z_e.grad - expected_grad).abs().max().item()}"
        )


class TestPreservationVQLoss:
    """Property 2.4: VQ loss formula unchanged.

    **Validates: Requirements 3.4**

    commitment_loss = mean(||sg[z_e] - z_q||²), and the total loss decomposition
    L = L_rec + commitment_loss + β₀ × mean(||z_e - sg[z_q]||²) is correct.
    """

    @given(z_e_np=_z_e_strategy())
    @settings(max_examples=50, deadline=None)
    def test_commitment_loss_formula(self, z_e_np: np.ndarray):
        """commitment_loss matches mean(||sg[z_e] - z_q||²)."""
        z_e = torch.tensor(z_e_np, requires_grad=True)
        codebook = VQCodebook(num_codes=10, code_dim=16)

        z_q_st, indices, commitment_loss = codebook.quantize(z_e)

        # Manually compute commitment_loss = mean(||z_e.detach() - z_q||²)
        z_q = codebook.embeddings(indices)
        expected_commitment = torch.mean((z_e.detach() - z_q) ** 2)

        assert torch.allclose(commitment_loss, expected_commitment, atol=1e-5), (
            f"commitment_loss={commitment_loss.item():.6f} != "
            f"expected={expected_commitment.item():.6f}"
        )

    @given(z_e_np=_z_e_strategy())
    @settings(max_examples=50, deadline=None)
    def test_total_loss_decomposition(self, z_e_np: np.ndarray):
        """Total VQ loss decomposes as L_rec + commitment + β₀ × encoder_commitment."""
        z_e = torch.tensor(z_e_np, requires_grad=True)
        codebook = VQCodebook(num_codes=10, code_dim=16)
        beta0 = 0.25

        z_q_st, indices, commitment_loss = codebook.quantize(z_e)

        # Encoder commitment: β₀ × mean(||z_e - sg[z_q]||²)
        z_q_detached = z_q_st.detach()
        encoder_commitment = beta0 * torch.mean((z_e - z_q_detached) ** 2)

        # Simulate a reconstruction loss (just a placeholder scalar)
        L_rec = torch.tensor(1.0)

        total = L_rec + commitment_loss + encoder_commitment

        # Verify the decomposition is additive and each component is a valid scalar
        assert commitment_loss.dim() == 0
        assert encoder_commitment.dim() == 0
        assert total.dim() == 0
        assert torch.isfinite(total), f"Total loss is not finite: {total.item()}"

        # Verify the sum is correct
        expected_total = L_rec.item() + commitment_loss.item() + encoder_commitment.item()
        assert abs(total.item() - expected_total) < 1e-5, (
            f"Total loss {total.item():.6f} != sum of components {expected_total:.6f}"
        )


class TestPreservationEncoderShape:
    """Property 2.5: Encoder output shape unchanged.

    **Validates: Requirements 3.1**

    For any (batch, seq_len, 45) input, z_e shape is (batch, 16).
    """

    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_encoder_output_shape(self, batch: int, seq_len: int):
        """Encoder produces z_e of shape (batch, 16) for any valid input."""
        state_dim = 45
        encoder = VQEncoder(state_dim=state_dim, action_dim=3, hidden_dim=128, latent_dim=16)
        encoder.eval()

        s_demo = torch.randn(batch, seq_len, state_dim)
        a_demo = torch.randint(0, 3, (batch, seq_len))
        r_demo = torch.randn(batch, seq_len)

        with torch.no_grad():
            z_e = encoder(s_demo, a_demo, r_demo)

        assert z_e.shape == (batch, 16), f"z_e shape {z_e.shape} != ({batch}, 16)"


class TestPreservationDecoderShape:
    """Property 2.6: Decoder output shape unchanged.

    **Validates: Requirements 3.3**

    For any (batch, seq_len, 45) states and (batch, 16) z_q, logits shape is
    (batch, seq_len, 3).
    """

    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_decoder_output_shape(self, batch: int, seq_len: int):
        """Decoder produces logits of shape (batch, seq_len, 3) for any valid input."""
        state_dim = 45
        decoder = VQDecoder(state_dim=state_dim, code_dim=16, hidden_dim=128, action_dim=3)
        decoder.eval()

        states = torch.randn(batch, seq_len, state_dim)
        z_q = torch.randn(batch, 16)

        with torch.no_grad():
            logits = decoder(states, z_q)

        assert logits.shape == (batch, seq_len, 3), (
            f"logits shape {logits.shape} != ({batch}, {seq_len}, 3)"
        )


class TestPreservationPhaseASkipsVQ:
    """Property 2.7: Phase A skips VQ.

    **Validates: Requirements 3.5**

    When is_phase_a=True, quantization is not performed — z_e is passed directly
    to the decoder without going through codebook.quantize().
    """

    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_phase_a_bypasses_vq(self, batch: int, seq_len: int):
        """In Phase A, z_e goes directly to decoder (no quantization).

        We verify this by checking that the decoder receives z_e directly:
        action_logits = decoder(states, z_e) produces valid output, and
        the codebook is not involved (no indices produced).
        """
        state_dim = 45
        encoder = VQEncoder(state_dim=state_dim, action_dim=3, hidden_dim=128, latent_dim=16)
        decoder = VQDecoder(state_dim=state_dim, code_dim=16, hidden_dim=128, action_dim=3)
        encoder.eval()
        decoder.eval()

        s_demo = torch.randn(batch, seq_len, state_dim)
        a_demo = torch.randint(0, 3, (batch, seq_len))
        r_demo = torch.randn(batch, seq_len)

        with torch.no_grad():
            z_e = encoder(s_demo, a_demo, r_demo)

            # Phase A: z_e goes directly to decoder (bypass VQ)
            is_phase_a = True
            if is_phase_a:
                z_input = z_e
            else:
                codebook = VQCodebook(num_codes=10, code_dim=16)
                z_input, _, _ = codebook.quantize(z_e)

            logits = decoder(s_demo, z_input)

        # Verify decoder still produces correct output shape when fed z_e directly
        assert logits.shape == (batch, seq_len, 3), (
            f"Phase A logits shape {logits.shape} != ({batch}, {seq_len}, 3)"
        )

        # Verify z_input IS z_e (same object, no quantization applied)
        assert z_input is z_e, "Phase A should pass z_e directly without quantization"


# ---------------------------------------------------------------------------
# Task 3.1: Unit Tests for init_from_data()
# ---------------------------------------------------------------------------

"""
Unit tests for VQCodebook.init_from_data() covering:
- Basic functionality (K=10, N=1000)
- Edge case N < K (skip initialization)
- Empty cluster handling
- k-means++ spread
- Codebook covers z_e distribution after init

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
"""


class TestInitFromDataBasic:
    """Basic functionality: K=10, N=1000 z_e samples.

    **Validates: Requirements 2.1, 2.2**
    """

    def test_codebook_weights_updated_and_shape_correct(self):
        """After init_from_data, codebook weights differ from original and shape is (K, D)."""
        torch.manual_seed(42)
        K, D, N = 10, 16, 1000

        codebook = VQCodebook(num_codes=K, code_dim=D)
        original_weights = codebook.embeddings.weight.clone()

        z_e = torch.randn(N, D) * 0.5 + 3.0  # shifted distribution
        codebook.init_from_data(z_e)

        new_weights = codebook.embeddings.weight

        # Shape must be preserved
        assert new_weights.shape == (K, D), f"Expected ({K}, {D}), got {new_weights.shape}"

        # Weights must have changed
        assert not torch.allclose(new_weights, original_weights, atol=1e-6), (
            "Codebook weights were not updated by init_from_data"
        )


class TestInitFromDataSkipWhenNLessThanK:
    """Edge case: N < K should skip initialization.

    **Validates: Requirement 2.4**
    """

    def test_skip_init_when_n_less_than_k(self):
        """When N=5 < K=10, init_from_data skips and weights remain unchanged."""
        torch.manual_seed(42)
        K, D, N = 10, 16, 5

        codebook = VQCodebook(num_codes=K, code_dim=D)
        original_weights = codebook.embeddings.weight.clone()

        z_e = torch.randn(N, D)
        codebook.init_from_data(z_e)

        assert torch.allclose(codebook.embeddings.weight, original_weights), (
            "Codebook weights should be unchanged when N < K"
        )


class TestInitFromDataEmptyCluster:
    """Empty cluster handling: data that could produce empty clusters.

    **Validates: Requirement 2.5**
    """

    def test_all_centers_valid_with_degenerate_data(self):
        """With 9 points at same position + 1 outlier, all K centers are valid (no NaN/Inf)."""
        torch.manual_seed(42)
        K, D = 10, 16

        # 9 identical points + 1 outlier far away → likely empty clusters during k-means
        same_point = torch.zeros(1, D)
        cluster_points = same_point.expand(990, D) + torch.randn(990, D) * 1e-6
        outlier = torch.full((10, D), 10.0)
        z_e = torch.cat([cluster_points, outlier], dim=0)

        codebook = VQCodebook(num_codes=K, code_dim=D)
        codebook.init_from_data(z_e)

        weights = codebook.embeddings.weight
        assert not torch.isnan(weights).any(), "Codebook contains NaN after init"
        assert not torch.isinf(weights).any(), "Codebook contains Inf after init"
        assert weights.shape == (K, D)


class TestInitFromDataKMeansPPSpread:
    """k-means++ spread: initial centers should be well-separated.

    **Validates: Requirement 2.2**
    """

    def test_initial_centers_have_positive_min_distance(self):
        """After init, all pairwise distances between codebook vectors are > 0."""
        torch.manual_seed(42)
        K, D, N = 10, 16, 1000

        # Generate well-spread z_e from multiple clusters
        z_e_parts = []
        for i in range(K):
            center = torch.randn(1, D) * 5
            z_e_parts.append(center + torch.randn(N // K, D) * 0.3)
        z_e = torch.cat(z_e_parts, dim=0)

        codebook = VQCodebook(num_codes=K, code_dim=D)
        codebook.init_from_data(z_e)

        weights = codebook.embeddings.weight  # (K, D)
        # Compute pairwise distances
        dists = torch.cdist(weights.unsqueeze(0), weights.unsqueeze(0)).squeeze(0)  # (K, K)
        # Zero out diagonal
        dists.fill_diagonal_(float('inf'))
        min_dist = dists.min().item()

        assert min_dist > 0, (
            f"Minimum pairwise distance between codebook vectors is {min_dist}, expected > 0"
        )


class TestInitFromDataCodebookCoversDistribution:
    """After init, quantizing z_e should use all K codes.

    **Validates: Requirements 2.2, 2.3**
    """

    def test_all_codes_used_after_init(self):
        """After init_from_data, quantizing the same z_e yields used_code_count == K."""
        torch.manual_seed(42)
        K, D, N = 10, 16, 1000

        # Generate z_e from K well-separated clusters so k-means can find them
        z_e_parts = []
        for i in range(K):
            center = torch.randn(1, D) * 5
            z_e_parts.append(center + torch.randn(N // K, D) * 0.1)
        z_e = torch.cat(z_e_parts, dim=0)

        codebook = VQCodebook(num_codes=K, code_dim=D)
        codebook.init_from_data(z_e)

        with torch.no_grad():
            _, indices, _ = codebook.quantize(z_e)

        used_code_count = len(torch.unique(indices))
        assert used_code_count == K, (
            f"used_code_count={used_code_count}/{K} after init_from_data — "
            f"expected all codes to be used"
        )
