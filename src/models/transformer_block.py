"""
Transformer Encoder Block for the Weather Transformer.

Combines Multi-Head Self-Attention with a Feed-Forward MLP,
using pre-norm architecture (LayerNorm before each sub-layer)
and residual connections.

Architecture (Pre-Norm, like modern ViTs):
    x → LayerNorm → MHSA → + (residual) → LayerNorm → MLP → + (residual) → output
"""

import torch
import torch.nn as nn

from src.models.attention import MultiHeadSelfAttention


class MLP(nn.Module):
    """
    Feed-Forward MLP block used inside the Transformer.
    
    Two linear layers with GELU activation in between.
    
    Args:
        embed_dim: Input and output dimension.
        mlp_ratio: Hidden dimension multiplier (hidden_dim = embed_dim * mlp_ratio).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Encoder Block with pre-norm architecture.
    
    Architecture:
        x ─→ LN ─→ MHSA ─→ (+) ─→ LN ─→ MLP ─→ (+) ─→ output
        │                    ↑     │                ↑
        └────── residual ────┘     └── residual ────┘
    
    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier.
        dropout: Dropout rate for attention and MLP.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Pre-norm layer for attention
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Pre-norm layer for MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-Forward MLP
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (B, N, embed_dim).
            return_attention: If True, also return attention weights.
            
        Returns:
            output: Tensor of shape (B, N, embed_dim).
            attn_weights: (optional) Tensor of shape (B, num_heads, N, N).
        """
        # Sub-layer 1: LayerNorm → MHSA → Residual
        residual = x
        x_norm = self.norm1(x)

        if return_attention:
            attn_out, attn_weights = self.attention(
                x_norm, return_attention=True
            )
        else:
            attn_out = self.attention(x_norm)

        x = residual + attn_out

        # Sub-layer 2: LayerNorm → MLP → Residual
        residual = x
        x = residual + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn_weights

        return x


if __name__ == "__main__":
    print("Transformer Block — Sanity Check")
    print("=" * 40)

    B, N, D = 2, 128, 256

    block = TransformerBlock(
        embed_dim=D, num_heads=8, mlp_ratio=4.0, dropout=0.0
    )

    x = torch.randn(B, N, D)
    out = block(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    # With attention weights
    out, attn = block(x, return_attention=True)
    print(f"Attention: {attn.shape}")

    # Verify residual connection works (output ≠ 0 for zero input)
    zero_input = torch.zeros(1, N, D)
    zero_out = block(zero_input)
    print(f"Zero input → output norm: {zero_out.norm().item():.4f} (should be > 0)")

    n_params = sum(p.numel() for p in block.parameters())
    print(f"Parameters: {n_params:,}")

    print(f"\n✅ Transformer block working!")
