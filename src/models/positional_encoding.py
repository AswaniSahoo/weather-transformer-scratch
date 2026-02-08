"""
Positional Encoding for the Weather Transformer.

Adds spatial position information to patch embeddings so the model
knows WHERE each patch is on the lat/lon grid.

Two options provided:
1. Learnable positional embeddings (default, like ViT)
2. Sinusoidal positional embeddings (fixed, like original Transformer)
"""

import math

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings for spatial patches.
    
    Each patch position gets a learned embedding vector that is
    added to the patch embedding. This is the standard ViT approach.
    
    Args:
        n_patches: Total number of patches (N = n_patches_h * n_patches_w).
        embed_dim: Dimension of each embedding vector.
        dropout: Dropout rate applied after adding position info.
    """

    def __init__(
        self,
        n_patches: int = 128,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_patches = n_patches
        self.embed_dim = embed_dim

        # Learnable position embeddings: one vector per patch position
        self.position_embeddings = nn.Parameter(
            torch.randn(1, n_patches, embed_dim) * 0.02
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to patch embeddings.
        
        Args:
            x: Patch embeddings of shape (B, N_patches, embed_dim).
            
        Returns:
            Position-encoded embeddings of shape (B, N_patches, embed_dim).
        """
        B, N, D = x.shape

        assert N == self.n_patches, (
            f"Expected {self.n_patches} patches, got {N}"
        )
        assert D == self.embed_dim, (
            f"Expected embed_dim {self.embed_dim}, got {D}"
        )

        # Add positional embeddings (broadcast across batch)
        x = x + self.position_embeddings

        x = self.dropout(x)

        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    
    Uses sin/cos functions at different frequencies to encode position,
    following the original "Attention is All You Need" paper.
    This variant encodes 2D spatial positions (lat, lon) into the
    embedding space.
    
    Args:
        n_patches: Total number of patches.
        embed_dim: Dimension of each embedding vector.
        n_patches_h: Number of patches along height (latitude).
        n_patches_w: Number of patches along width (longitude).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_patches: int = 128,
        embed_dim: int = 256,
        n_patches_h: int = 8,
        n_patches_w: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_patches = n_patches
        self.embed_dim = embed_dim

        # Build 2D sinusoidal encoding
        pe = self._build_2d_sinusoidal(n_patches_h, n_patches_w, embed_dim)

        # Register as buffer (not a parameter — won't be trained)
        self.register_buffer("position_embeddings", pe)

        self.dropout = nn.Dropout(p=dropout)

    def _build_2d_sinusoidal(
        self, n_h: int, n_w: int, embed_dim: int
    ) -> torch.Tensor:
        """
        Build 2D sinusoidal positional encoding.
        
        Splits embed_dim in half: first half encodes height (lat),
        second half encodes width (lon).
        
        Returns:
            Tensor of shape (1, n_h * n_w, embed_dim).
        """
        assert embed_dim % 2 == 0, "embed_dim must be even for 2D sinusoidal PE"

        half_dim = embed_dim // 2

        # Height (latitude) encoding
        pe_h = self._sinusoidal_1d(n_h, half_dim)  # (n_h, half_dim)

        # Width (longitude) encoding
        pe_w = self._sinusoidal_1d(n_w, half_dim)  # (n_w, half_dim)

        # Combine: for each (h, w) position, concatenate pe_h[h] and pe_w[w]
        pe = torch.zeros(n_h, n_w, embed_dim)
        for h in range(n_h):
            for w in range(n_w):
                pe[h, w, :half_dim] = pe_h[h]
                pe[h, w, half_dim:] = pe_w[w]

        # Reshape to (1, N_patches, embed_dim)
        pe = pe.reshape(1, n_h * n_w, embed_dim)

        return pe

    def _sinusoidal_1d(self, n_positions: int, dim: int) -> torch.Tensor:
        """
        Standard 1D sinusoidal encoding.
        
        Args:
            n_positions: Number of positions.
            dim: Encoding dimension.
            
        Returns:
            Tensor of shape (n_positions, dim).
        """
        pe = torch.zeros(n_positions, dim)
        position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional encoding to patch embeddings.
        
        Args:
            x: Patch embeddings of shape (B, N_patches, embed_dim).
            
        Returns:
            Position-encoded embeddings of shape (B, N_patches, embed_dim).
        """
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    print("Positional Encoding — Sanity Check")
    print("=" * 40)

    B, N, D = 2, 128, 256

    x = torch.randn(B, N, D)

    # Test learnable
    learnable_pe = LearnablePositionalEncoding(n_patches=N, embed_dim=D)
    out_l = learnable_pe(x)
    print(f"Learnable PE — Input: {x.shape} → Output: {out_l.shape}")

    # Test sinusoidal
    sinusoidal_pe = SinusoidalPositionalEncoding(
        n_patches=N, embed_dim=D, n_patches_h=8, n_patches_w=16
    )
    out_s = sinusoidal_pe(x)
    print(f"Sinusoidal PE — Input: {x.shape} → Output: {out_s.shape}")

    # Verify different positions get different encodings
    pos_emb = learnable_pe.position_embeddings[0]  # (N, D)
    diff = (pos_emb[0] - pos_emb[1]).abs().sum().item()
    print(f"\nDifference between position 0 and 1: {diff:.4f} (should be > 0)")

    print("\n✅ Both positional encodings working!")
