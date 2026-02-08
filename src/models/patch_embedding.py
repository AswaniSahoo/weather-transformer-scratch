"""
Patch Embedding layer for the Weather Transformer.

Splits a (B, C, H, W) weather grid into non-overlapping patches
and projects each patch into an embedding vector.

Example:
    Input:  (B, 4, 32, 64)  — 4 variables on a 32×64 grid
    Patch:  4×4 pixels
    Output: (B, 128, embed_dim) — 128 patch embeddings
    
    N_patches = (H // P) * (W // P) = (32//4) * (64//4) = 8 * 16 = 128
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbedding(nn.Module):
    """
    Convert a 2D weather grid into a sequence of patch embeddings.
    
    Uses a Conv2d with kernel_size=patch_size and stride=patch_size to
    extract non-overlapping patches and project them in one step.
    This is mathematically equivalent to unfold + linear, but more efficient.
    
    Args:
        in_channels: Number of input channels (weather variables).
        embed_dim: Dimension of each patch embedding vector.
        patch_size: Size of each square patch (P×P pixels).
        img_height: Height of the input grid.
        img_width: Width of the input grid.
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        patch_size: int = 4,
        img_height: int = 32,
        img_width: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width

        # Validate that image dimensions are divisible by patch size
        assert img_height % patch_size == 0, (
            f"Image height {img_height} must be divisible by patch_size {patch_size}"
        )
        assert img_width % patch_size == 0, (
            f"Image width {img_width} must be divisible by patch_size {patch_size}"
        )

        # Number of patches in each dimension and total
        self.n_patches_h = img_height // patch_size
        self.n_patches_w = img_width // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        # Patch size flattened: each patch has C * P * P values
        self.patch_dim = in_channels * patch_size * patch_size

        # Projection: Conv2d acts as a sliding window extractor + linear projection
        # kernel_size = patch_size, stride = patch_size → non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Layer normalization on the embedding dimension
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract patches and project to embeddings.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Patch embeddings of shape (B, N_patches, embed_dim).
        """
        B, C, H, W = x.shape

        assert C == self.in_channels, (
            f"Expected {self.in_channels} channels, got {C}"
        )
        assert H == self.img_height and W == self.img_width, (
            f"Expected ({self.img_height}, {self.img_width}), got ({H}, {W})"
        )

        # Conv2d: (B, C, H, W) → (B, embed_dim, n_patches_h, n_patches_w)
        x = self.projection(x)

        # Reshape: (B, embed_dim, n_h, n_w) → (B, embed_dim, N_patches)
        x = x.flatten(2)

        # Transpose: (B, embed_dim, N_patches) → (B, N_patches, embed_dim)
        x = x.transpose(1, 2)

        # Normalize
        x = self.norm(x)

        return x

    def get_patch_grid_shape(self) -> Tuple[int, int]:
        """Return the (n_patches_h, n_patches_w) grid shape."""
        return (self.n_patches_h, self.n_patches_w)


if __name__ == "__main__":
    # Quick sanity check
    print("PatchEmbedding — Sanity Check")
    print("=" * 40)

    patch_emb = PatchEmbedding(
        in_channels=4,
        embed_dim=256,
        patch_size=4,
        img_height=32,
        img_width=64,
    )

    x = torch.randn(2, 4, 32, 64)
    out = patch_emb(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"N patches:    {patch_emb.n_patches}")
    print(f"Patch grid:   {patch_emb.get_patch_grid_shape()}")
    print(f"Parameters:   {sum(p.numel() for p in patch_emb.parameters()):,}")
    print(f"\n✅ Expected output shape: (2, 128, 256) — Got: {tuple(out.shape)}")
