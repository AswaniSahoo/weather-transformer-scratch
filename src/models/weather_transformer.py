"""
Full Weather Transformer model.

Assembles all components into an end-to-end model:
    Input (B, C, H, W) → Patch Embedding → + Positional Encoding
    → N × Transformer Blocks → LayerNorm → Prediction Head → Output (B, C, H, W)

Predicts the next weather state (t+6h) from the current state.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.models.patch_embedding import PatchEmbedding
from src.models.positional_encoding import LearnablePositionalEncoding
from src.models.transformer_block import TransformerBlock


class WeatherTransformer(nn.Module):
    """
    Vision Transformer for weather forecasting.
    
    Takes a weather state grid (B, C, H, W) and predicts the next
    timestep weather state (B, C, H, W).
    
    Args:
        in_channels: Number of input weather variables.
        out_channels: Number of output weather variables.
        img_height: Height of the weather grid.
        img_width: Width of the weather grid.
        patch_size: Size of each square patch.
        embed_dim: Transformer embedding dimension.
        num_heads: Number of attention heads per block.
        num_layers: Number of transformer encoder blocks.
        mlp_ratio: MLP hidden dimension = embed_dim * mlp_ratio.
        dropout: Dropout rate throughout the model.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_height: int = 32,
        img_width: int = 64,
        patch_size: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # --- Step 1: Patch Embedding ---
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_height=img_height,
            img_width=img_width,
        )

        n_patches = self.patch_embedding.n_patches
        self.n_patches = n_patches

        # --- Step 2: Positional Encoding ---
        self.positional_encoding = LearnablePositionalEncoding(
            n_patches=n_patches,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # --- Step 3: Transformer Encoder Blocks ---
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # --- Step 4: Final LayerNorm ---
        self.norm = nn.LayerNorm(embed_dim)

        # --- Step 5: Prediction Head ---
        # Project each patch embedding back to patch pixel space
        # embed_dim → patch_size * patch_size * out_channels
        self.prediction_head = nn.Linear(
            embed_dim,
            patch_size * patch_size * out_channels,
        )

        # Store grid info for reshaping
        self.n_patches_h = self.patch_embedding.n_patches_h
        self.n_patches_w = self.patch_embedding.n_patches_w

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ):
        """
        Forward pass: weather grid → predicted weather grid.
        
        Args:
            x: Input weather state, shape (B, C, H, W).
            return_attention: If True, return attention weights from all layers.
            
        Returns:
            output: Predicted weather state, shape (B, C_out, H, W).
            attention_weights: (optional) List of attention weight tensors.
        """
        B = x.shape[0]
        attention_weights = []

        # Step 1: Patch embedding (B, C, H, W) → (B, N, embed_dim)
        x = self.patch_embedding(x)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Pass through transformer blocks
        for block in self.transformer_blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = block(x)

        # Step 4: Final layer norm
        x = self.norm(x)

        # Step 5: Prediction head (B, N, embed_dim) → (B, N, P*P*C_out)
        x = self.prediction_head(x)

        # Step 6: Reshape back to image grid
        # (B, N, P*P*C_out) → (B, C_out, H, W)
        x = self._patches_to_image(x, B)

        if return_attention:
            return x, attention_weights

        return x

    def _patches_to_image(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reshape patch predictions back to a full image grid.
        
        Args:
            x: Patch predictions, shape (B, N_patches, P*P*C_out).
            batch_size: Batch size.
            
        Returns:
            Image tensor of shape (B, C_out, H, W).
        """
        P = self.patch_size
        C = self.out_channels
        nH = self.n_patches_h
        nW = self.n_patches_w

        # (B, N, P*P*C) → (B, nH, nW, P, P, C)
        x = x.view(batch_size, nH, nW, P, P, C)

        # (B, nH, nW, P, P, C) → (B, C, nH, P, nW, P)
        x = x.permute(0, 5, 1, 3, 2, 4)

        # (B, C, nH, P, nW, P) → (B, C, H, W)
        x = x.contiguous().view(batch_size, C, nH * P, nW * P)

        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict:
        """Return model configuration as a dict."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_height": self.img_height,
            "img_width": self.img_width,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "n_patches": self.n_patches,
            "total_params": self.count_parameters(),
        }


if __name__ == "__main__":
    print("Weather Transformer — Full Model Sanity Check")
    print("=" * 50)

    model = WeatherTransformer(
        in_channels=4,
        out_channels=4,
        img_height=32,
        img_width=64,
        patch_size=4,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.1,
    )

    x = torch.randn(2, 4, 32, 64)
    out = model(x)

    print(f"Input:      {x.shape}")
    print(f"Output:     {out.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Config:     {model.get_config()}")

    # Test backward pass
    loss = out.sum()
    loss.backward()
    print(f"Backward pass: ✅")

    # Test with attention weights
    out, attns = model(x, return_attention=True)
    print(f"Attention layers returned: {len(attns)}")
    print(f"Each attention shape: {attns[0].shape}")

    print(f"\n✅ Full model working! Shape: {x.shape} → {out.shape}")
