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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
