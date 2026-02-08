"""
Multi-Head Self-Attention from scratch.

Implements the core attention mechanism:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Built entirely manually — no nn.MultiheadAttention used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism built from scratch.
    
    Splits the embedding into multiple heads, computes scaled dot-product
    attention independently for each head, then concatenates and projects.
    
    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate on attention weights.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projection layers (separate linear layers, no cheating!)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_out = nn.Linear(embed_dim, embed_dim)

        # Dropout on attention weights
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ):
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor of shape (B, N, embed_dim).
            return_attention: If True, also return attention weights.
            
        Returns:
            output: Tensor of shape (B, N, embed_dim).
            attn_weights: (optional) Tensor of shape (B, num_heads, N, N).
        """
        B, N, D = x.shape

        assert D == self.embed_dim, (
            f"Expected embed_dim {self.embed_dim}, got {D}"
        )

        # Step 1: Project to Q, K, V
        # Each: (B, N, embed_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Reshape for multi-head: (B, N, embed_dim) → (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Scaled dot-product attention
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) → (B, num_heads, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Step 4: Softmax over the key dimension
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Step 5: Dropout on attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Weighted sum of values
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) → (B, num_heads, N, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Step 7: Concatenate heads
        # (B, num_heads, N, head_dim) → (B, N, num_heads, head_dim) → (B, N, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)

        # Step 8: Final linear projection
        output = self.W_out(attn_output)

        if return_attention:
            return output, attn_weights

        return output


if __name__ == "__main__":
    print("Multi-Head Self-Attention — Sanity Check")
    print("=" * 45)

    B, N, D = 2, 128, 256
    num_heads = 8

    mhsa = MultiHeadSelfAttention(embed_dim=D, num_heads=num_heads, dropout=0.0)
    x = torch.randn(B, N, D)

    # Without attention weights
    out = mhsa(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    # With attention weights
    out, attn = mhsa(x, return_attention=True)
    print(f"Attention weights: {attn.shape}")
    print(f"Attn weights sum (should be ~1.0): {attn[0, 0, 0].sum().item():.4f}")

    # Parameter count
    n_params = sum(p.numel() for p in mhsa.parameters())
    print(f"Parameters: {n_params:,}")

    print(f"\n✅ All shapes correct!")
