"""
Unit tests for model components.

Tests each transformer building block independently.
Run with: pytest tests/test_model.py -v
"""

import pytest
import torch

from src.models.patch_embedding import PatchEmbedding


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
