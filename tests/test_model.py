"""
Unit tests for model components.

Tests each transformer building block independently.
Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn

from src.models.patch_embedding import PatchEmbedding
from src.models.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
)
from src.models.attention import MultiHeadSelfAttention
from src.models.transformer_block import TransformerBlock, MLP
from src.models.weather_transformer import WeatherTransformer
from src.models.physics_loss import (
    PhysicsInformedLoss,
    SpatialSmoothnessLoss,
    ConservationLoss,
)


# ============================================================
# Test Constants
# ============================================================
BATCH_SIZE = 4
IN_CHANNELS = 4
EMBED_DIM = 256
PATCH_SIZE = 4
IMG_HEIGHT = 32
IMG_WIDTH = 64
N_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)  # 128


# ============================================================
# PatchEmbedding Tests
# ============================================================
class TestPatchEmbedding:
    """Tests for PatchEmbedding module."""

    @pytest.fixture
    def patch_emb(self):
        """Create a PatchEmbedding instance."""
        return PatchEmbedding(
            in_channels=IN_CHANNELS,
            embed_dim=EMBED_DIM,
            patch_size=PATCH_SIZE,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
        )

    @pytest.fixture
    def sample_input(self):
        """Create a sample input tensor."""
        return torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    def test_output_shape(self, patch_emb, sample_input):
        """Test that output shape is (B, N_patches, embed_dim)."""
        output = patch_emb(sample_input)
        expected_shape = (BATCH_SIZE, N_PATCHES, EMBED_DIM)
        assert output.shape == expected_shape, (
            f"Expected {expected_shape}, got {output.shape}"
        )

    def test_n_patches_calculation(self, patch_emb):
        """Test that number of patches is calculated correctly."""
        assert patch_emb.n_patches == N_PATCHES
        assert patch_emb.n_patches_h == IMG_HEIGHT // PATCH_SIZE
        assert patch_emb.n_patches_w == IMG_WIDTH // PATCH_SIZE

    def test_output_dtype(self, patch_emb, sample_input):
        """Test that output is float32."""
        output = patch_emb(sample_input)
        assert output.dtype == torch.float32

    def test_batch_independence(self, patch_emb):
        """Test that different batch elements produce different outputs."""
        x = torch.randn(2, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        output = patch_emb(x)
        assert not torch.allclose(output[0], output[1]), (
            "Different inputs should produce different outputs"
        )

    def test_gradient_flow(self, patch_emb, sample_input):
        """Test that gradients flow through the module."""
        sample_input.requires_grad_(True)
        output = patch_emb(sample_input)
        loss = output.sum()
        loss.backward()
        assert sample_input.grad is not None, "Gradients should flow to input"
        assert sample_input.grad.shape == sample_input.shape

    def test_patch_grid_shape(self, patch_emb):
        """Test the get_patch_grid_shape helper."""
        grid_shape = patch_emb.get_patch_grid_shape()
        assert grid_shape == (8, 16)  # 32//4, 64//4

    def test_wrong_channels_raises(self, patch_emb):
        """Test that wrong number of channels raises assertion."""
        wrong_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)  # 3 instead of 4
        with pytest.raises(AssertionError):
            patch_emb(wrong_input)

    def test_wrong_spatial_dims_raises(self, patch_emb):
        """Test that wrong spatial dimensions raise assertion."""
        wrong_input = torch.randn(1, IN_CHANNELS, 16, 16)  # wrong H, W
        with pytest.raises(AssertionError):
            patch_emb(wrong_input)

    def test_indivisible_patch_size_raises(self):
        """Test that non-divisible patch size raises assertion."""
        with pytest.raises(AssertionError):
            PatchEmbedding(
                in_channels=4,
                embed_dim=256,
                patch_size=5,  # 32 % 5 != 0
                img_height=32,
                img_width=64,
            )

    def test_parameter_count(self, patch_emb):
        """Test that parameter count is reasonable."""
        n_params = sum(p.numel() for p in patch_emb.parameters())
        # Conv2d: in_channels * embed_dim * patch_size^2 + embed_dim (bias)
        # LayerNorm: 2 * embed_dim (weight + bias)
        expected_conv = IN_CHANNELS * EMBED_DIM * PATCH_SIZE * PATCH_SIZE + EMBED_DIM
        expected_norm = 2 * EMBED_DIM
        expected_total = expected_conv + expected_norm
        assert n_params == expected_total, (
            f"Expected {expected_total} params, got {n_params}"
        )




# ============================================================
# Positional Encoding Tests
# ============================================================
class TestLearnablePositionalEncoding:
    """Tests for LearnablePositionalEncoding."""

    @pytest.fixture
    def learnable_pe(self):
        return LearnablePositionalEncoding(
            n_patches=N_PATCHES, embed_dim=EMBED_DIM, dropout=0.0
        )

    @pytest.fixture
    def pe_input(self):
        return torch.randn(BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_output_shape(self, learnable_pe, pe_input):
        """Output shape must match input shape."""
        output = learnable_pe(pe_input)
        assert output.shape == (BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_output_differs_from_input(self, learnable_pe, pe_input):
        """Adding positional info should change the values."""
        output = learnable_pe(pe_input)
        assert not torch.allclose(output, pe_input)

    def test_different_positions_different_encodings(self, learnable_pe):
        """Each position should have a unique encoding vector."""
        pos_emb = learnable_pe.position_embeddings[0]  # (N, D)
        assert not torch.allclose(pos_emb[0], pos_emb[1])

    def test_gradient_flow(self, learnable_pe, pe_input):
        """Gradients should flow through positional encoding."""
        pe_input.requires_grad_(True)
        output = learnable_pe(pe_input)
        output.sum().backward()
        assert pe_input.grad is not None

    def test_position_embeddings_are_learned(self, learnable_pe):
        """Position embeddings should be nn.Parameter (trainable)."""
        assert isinstance(learnable_pe.position_embeddings, nn.Parameter)


class TestSinusoidalPositionalEncoding:
    """Tests for SinusoidalPositionalEncoding."""

    @pytest.fixture
    def sinusoidal_pe(self):
        return SinusoidalPositionalEncoding(
            n_patches=N_PATCHES, embed_dim=EMBED_DIM,
            n_patches_h=8, n_patches_w=16, dropout=0.0,
        )

    @pytest.fixture
    def pe_input(self):
        return torch.randn(BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_output_shape(self, sinusoidal_pe, pe_input):
        """Output shape must match input shape."""
        output = sinusoidal_pe(pe_input)
        assert output.shape == (BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_encoding_is_fixed(self, sinusoidal_pe):
        """Sinusoidal encoding should be a buffer, not a parameter."""
        assert "position_embeddings" in dict(sinusoidal_pe.named_buffers())
        assert "position_embeddings" not in dict(sinusoidal_pe.named_parameters())

    def test_different_positions_different_encodings(self, sinusoidal_pe):
        """Different spatial positions should have different encodings."""
        pe = sinusoidal_pe.position_embeddings[0]  # (N, D)
        assert not torch.allclose(pe[0], pe[1])


# ============================================================
# Multi-Head Self-Attention Tests
# ============================================================
NUM_HEADS = 8


class TestMultiHeadSelfAttention:
    """Tests for MultiHeadSelfAttention module."""

    @pytest.fixture
    def mhsa(self):
        return MultiHeadSelfAttention(
            embed_dim=EMBED_DIM, num_heads=NUM_HEADS, dropout=0.0
        )

    @pytest.fixture
    def attn_input(self):
        return torch.randn(BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_output_shape(self, mhsa, attn_input):
        """Output shape must match input shape (B, N, D)."""
        output = mhsa(attn_input)
        assert output.shape == (BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_attention_weights_shape(self, mhsa, attn_input):
        """Attention weights should be (B, num_heads, N, N)."""
        _, attn_weights = mhsa(attn_input, return_attention=True)
        assert attn_weights.shape == (BATCH_SIZE, NUM_HEADS, N_PATCHES, N_PATCHES)

    def test_attention_weights_sum_to_one(self, mhsa, attn_input):
        """Each row of attention weights should sum to 1 (softmax)."""
        _, attn_weights = mhsa(attn_input, return_attention=True)
        row_sums = attn_weights.sum(dim=-1)  # (B, heads, N)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_gradient_flow(self, mhsa, attn_input):
        """Gradients should flow through attention."""
        attn_input.requires_grad_(True)
        output = mhsa(attn_input)
        output.sum().backward()
        assert attn_input.grad is not None

    def test_no_nn_multiheadattention(self, mhsa):
        """Verify we're NOT using PyTorch's built-in MHA."""
        for module in mhsa.modules():
            assert not isinstance(module, nn.MultiheadAttention), (
                "Must implement attention from scratch!"
            )

    def test_head_dim_calculation(self, mhsa):
        """head_dim should be embed_dim // num_heads."""
        assert mhsa.head_dim == EMBED_DIM // NUM_HEADS

    def test_indivisible_heads_raises(self):
        """embed_dim not divisible by num_heads should raise."""
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(embed_dim=256, num_heads=7)


# ============================================================
# Transformer Block Tests
# ============================================================
MLP_RATIO = 4.0


class TestMLP:
    """Tests for the MLP sub-module."""

    @pytest.fixture
    def mlp(self):
        return MLP(embed_dim=EMBED_DIM, mlp_ratio=MLP_RATIO, dropout=0.0)

    def test_output_shape(self, mlp):
        """MLP output shape must match input shape."""
        x = torch.randn(BATCH_SIZE, N_PATCHES, EMBED_DIM)
        assert mlp(x).shape == (BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_hidden_dim(self, mlp):
        """Hidden dimension should be embed_dim * mlp_ratio."""
        assert mlp.fc1.out_features == int(EMBED_DIM * MLP_RATIO)


class TestTransformerBlock:
    """Tests for TransformerBlock module."""

    @pytest.fixture
    def block(self):
        return TransformerBlock(
            embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO, dropout=0.0,
        )

    @pytest.fixture
    def block_input(self):
        return torch.randn(BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_output_shape(self, block, block_input):
        """Output shape must match input shape."""
        output = block(block_input)
        assert output.shape == (BATCH_SIZE, N_PATCHES, EMBED_DIM)

    def test_residual_connection(self, block):
        """Output should be non-zero even with zero input (due to bias terms)."""
        zero_input = torch.zeros(1, N_PATCHES, EMBED_DIM)
        output = block(zero_input)
        assert output.norm().item() > 0, "Residual connection should produce non-zero output"

    def test_output_changes_with_input(self, block):
        """Different inputs should produce different outputs."""
        x1 = torch.randn(1, N_PATCHES, EMBED_DIM)
        x2 = torch.randn(1, N_PATCHES, EMBED_DIM)
        out1 = block(x1)
        out2 = block(x2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flow(self, block, block_input):
        """Gradients should flow through the block."""
        block_input.requires_grad_(True)
        output = block(block_input)
        output.sum().backward()
        assert block_input.grad is not None

    def test_attention_weights_returned(self, block, block_input):
        """return_attention=True should return attention weights."""
        output, attn = block(block_input, return_attention=True)
        assert output.shape == (BATCH_SIZE, N_PATCHES, EMBED_DIM)
        assert attn.shape == (BATCH_SIZE, NUM_HEADS, N_PATCHES, N_PATCHES)

    def test_pre_norm_architecture(self, block):
        """Block should have two LayerNorm modules (pre-norm)."""
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)


# ============================================================
# Full Weather Transformer Tests
# ============================================================
NUM_LAYERS = 6


class TestWeatherTransformer:
    """Tests for the full WeatherTransformer model."""

    @pytest.fixture
    def model(self):
        return WeatherTransformer(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            mlp_ratio=MLP_RATIO,
            dropout=0.0,
        )

    @pytest.fixture
    def model_input(self):
        return torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    def test_output_shape(self, model, model_input):
        """Output must be same shape as input: (B, C, H, W)."""
        output = model(model_input)
        assert output.shape == (BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    def test_single_sample(self, model):
        """Model should work with batch_size=1."""
        x = torch.randn(1, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        out = model(x)
        assert out.shape == (1, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    def test_backward_pass(self, model, model_input):
        """Full backward pass should work without errors."""
        output = model(model_input)
        loss = output.sum()
        loss.backward()
        # Check that all parameters received gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_attention_weights_returned(self, model, model_input):
        """return_attention should give list of NUM_LAYERS attention tensors."""
        output, attns = model(model_input, return_attention=True)
        assert len(attns) == NUM_LAYERS
        for attn in attns:
            assert attn.shape == (BATCH_SIZE, NUM_HEADS, model.n_patches, model.n_patches)

    def test_parameter_count_reasonable(self, model):
        """Model should have a reasonable number of parameters."""
        n_params = model.count_parameters()
        # With embed_dim=256, 6 layers → should be roughly 3-5M params
        assert 1_000_000 < n_params < 20_000_000, (
            f"Parameter count {n_params:,} seems unreasonable"
        )

    def test_get_config(self, model):
        """get_config should return all key architecture details."""
        config = model.get_config()
        assert config["in_channels"] == IN_CHANNELS
        assert config["embed_dim"] == EMBED_DIM
        assert config["num_layers"] == NUM_LAYERS
        assert "total_params" in config

    def test_different_inputs_different_outputs(self, model):
        """Model should produce different outputs for different inputs."""
        x1 = torch.randn(1, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        x2 = torch.randn(1, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        assert not torch.allclose(model(x1), model(x2))

    def test_model_is_differentiable(self, model, model_input):
        """Input gradients should flow through the full model."""
        model_input.requires_grad_(True)
        output = model(model_input)
        output.sum().backward()
        assert model_input.grad is not None
        assert model_input.grad.shape == model_input.shape


# ============================================================
# Physics-Informed Loss Tests
# ============================================================

class TestSpatialSmoothnessLoss:
    """Tests for SpatialSmoothnessLoss."""

    def test_smooth_input_low_loss(self):
        """A perfectly uniform field should have near-zero smoothness loss."""
        smooth = torch.ones(2, 4, 32, 64) * 5.0
        loss = SpatialSmoothnessLoss()(smooth)
        assert loss.item() < 1e-6

    def test_noisy_input_higher_loss(self):
        """A noisy field should have higher smoothness loss than a smooth one."""
        smooth = torch.ones(2, 4, 32, 64)
        noisy = torch.randn(2, 4, 32, 64)
        loss_fn = SpatialSmoothnessLoss()
        assert loss_fn(noisy).item() > loss_fn(smooth).item()

    def test_gradient_flow(self):
        """Loss should be differentiable."""
        x = torch.randn(2, 4, 32, 64, requires_grad=True)
        loss = SpatialSmoothnessLoss()(x)
        loss.backward()
        assert x.grad is not None


class TestConservationLoss:
    """Tests for ConservationLoss."""

    def test_identical_means_zero_loss(self):
        """If pred and target have same global mean, loss should be ~0."""
        x = torch.randn(2, 4, 32, 64)
        loss = ConservationLoss()(x, x)
        assert loss.item() < 1e-6

    def test_different_means_nonzero_loss(self):
        """If pred and target have different means, loss should be > 0."""
        pred = torch.ones(2, 4, 32, 64) * 10.0
        target = torch.zeros(2, 4, 32, 64)
        loss = ConservationLoss()(pred, target)
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Loss should be differentiable."""
        pred = torch.randn(2, 4, 32, 64, requires_grad=True)
        target = torch.randn(2, 4, 32, 64)
        loss = ConservationLoss()(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestPhysicsInformedLoss:
    """Tests for combined PhysicsInformedLoss."""

    @pytest.fixture
    def loss_fn(self):
        return PhysicsInformedLoss(
            mse_weight=1.0, smoothness_weight=0.1, conservation_weight=0.05
        )

    def test_returns_dict(self, loss_fn):
        """Loss should return dict with total, mse, smoothness, conservation."""
        pred = torch.randn(2, 4, 32, 64)
        target = torch.randn(2, 4, 32, 64)
        result = loss_fn(pred, target)
        assert "total" in result
        assert "mse" in result
        assert "smoothness" in result
        assert "conservation" in result

    def test_total_is_weighted_sum(self, loss_fn):
        """Total should approximately equal weighted sum of components."""
        pred = torch.randn(2, 4, 32, 64)
        target = torch.randn(2, 4, 32, 64)
        result = loss_fn(pred, target)
        expected = (
            1.0 * result["mse"]
            + 0.1 * result["smoothness"]
            + 0.05 * result["conservation"]
        )
        assert torch.allclose(result["total"], expected, atol=1e-5)

    def test_gradient_flows_through_total(self, loss_fn):
        """Backward on total loss should produce gradients."""
        pred = torch.randn(2, 4, 32, 64, requires_grad=True)
        target = torch.randn(2, 4, 32, 64)
        result = loss_fn(pred, target)
        result["total"].backward()
        assert pred.grad is not None

    def test_zero_weights_disable_components(self):
        """Setting a weight to 0 should effectively disable that component."""
        loss_fn = PhysicsInformedLoss(
            mse_weight=1.0, smoothness_weight=0.0, conservation_weight=0.0
        )
        pred = torch.randn(2, 4, 32, 64)
        target = torch.randn(2, 4, 32, 64)
        result = loss_fn(pred, target)
        # Total should equal just MSE
        assert torch.allclose(result["total"], result["mse"], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
