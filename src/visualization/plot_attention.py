"""
Attention weight visualization.

Visualizes which spatial regions attend to each other in the
transformer's self-attention mechanism.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

import torch


def plot_attention_maps(
    attention_weights: torch.Tensor,
    query_positions: Optional[List[int]] = None,
    layer_idx: int = 0,
    head_idx: int = 0,
    n_patches_h: int = 8,
    n_patches_w: int = 16,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Visualize attention weights showing which patches attend to which.
    
    Args:
        attention_weights: Attention tensor (B, num_heads, N, N) or 
                          list of such tensors per layer.
        query_positions: List of query patch positions to visualize.
                        If None, uses a few representative positions.
        layer_idx: Which layer's attention to visualize.
        head_idx: Which attention head to visualize.
        n_patches_h: Number of patches along height.
        n_patches_w: Number of patches along width.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
    """
    # Handle list of attention weights (per layer)
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx]
    else:
        attn = attention_weights

    # Get attention for first batch item, specified head
    # Shape: (N, N)
    if attn.dim() == 4:
        attn_head = attn[0, head_idx].detach().cpu().numpy()
    else:
        attn_head = attn.detach().cpu().numpy()

    N = attn_head.shape[0]

    # Default query positions: corners and center
    if query_positions is None:
        center = N // 2
        query_positions = [
            0,                          # Top-left
            n_patches_w - 1,            # Top-right
            center,                     # Center
            N - n_patches_w,            # Bottom-left
            N - 1,                      # Bottom-right
        ]
        query_positions = [p for p in query_positions if 0 <= p < N]

    n_queries = len(query_positions)
    fig, axes = plt.subplots(2, n_queries, figsize=figsize)

    if n_queries == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Attention Visualization — Layer {layer_idx}, Head {head_idx}",
        fontsize=14, fontweight="bold"
    )

    for i, query_pos in enumerate(query_positions):
        # Get attention scores from this query position
        attn_from_query = attn_head[query_pos].reshape(n_patches_h, n_patches_w)

        # Mark query position
        query_h = query_pos // n_patches_w
        query_w = query_pos % n_patches_w

        # Top row: attention maps
        ax_top = axes[0, i]
        im = ax_top.imshow(attn_from_query, cmap="hot", vmin=0)
        ax_top.scatter([query_w], [query_h], c="cyan", s=100, marker="*", 
                      edgecolors="white", linewidths=2, zorder=10)
        ax_top.set_title(f"Query: ({query_h}, {query_w})")
        ax_top.set_xlabel("Patch W")
        ax_top.set_ylabel("Patch H")
        plt.colorbar(im, ax=ax_top, shrink=0.8)

        # Bottom row: top-k attended positions
        ax_bot = axes[1, i]
        top_k = 10
        attn_flat = attn_from_query.flatten()
        top_indices = np.argsort(attn_flat)[-top_k:][::-1]

        # Create mask showing top attended positions
        top_mask = np.zeros_like(attn_flat)
        for rank, idx in enumerate(top_indices):
            top_mask[idx] = top_k - rank
        top_mask = top_mask.reshape(n_patches_h, n_patches_w)

        ax_bot.imshow(top_mask, cmap="YlOrRd")
        ax_bot.scatter([query_w], [query_h], c="cyan", s=100, marker="*",
                      edgecolors="black", linewidths=2, zorder=10)
        ax_bot.set_title(f"Top-{top_k} Attended")
        ax_bot.set_xlabel("Patch W")
        ax_bot.set_ylabel("Patch H")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention maps to {save_path}")

    return fig


def plot_attention_heads_comparison(
    attention_weights: torch.Tensor,
    query_pos: int,
    n_patches_h: int = 8,
    n_patches_w: int = 16,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Compare attention patterns across all heads for a single query.
    
    Args:
        attention_weights: Attention tensor (B, num_heads, N, N).
        query_pos: Query patch position.
        n_patches_h: Number of patches along height.
        n_patches_w: Number of patches along width.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
    """
    attn = attention_weights[0].detach().cpu().numpy()  # (num_heads, N, N)
    num_heads = attn.shape[0]

    # Layout: 2 rows
    n_cols = (num_heads + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)

    query_h = query_pos // n_patches_w
    query_w = query_pos % n_patches_w

    fig.suptitle(
        f"Attention Heads Comparison — Query: ({query_h}, {query_w})",
        fontsize=14, fontweight="bold"
    )

    for head_idx in range(num_heads):
        row = head_idx // n_cols
        col = head_idx % n_cols
        ax = axes[row, col]

        attn_map = attn[head_idx, query_pos].reshape(n_patches_h, n_patches_w)
        im = ax.imshow(attn_map, cmap="hot", vmin=0)
        ax.scatter([query_w], [query_h], c="cyan", s=80, marker="*",
                  edgecolors="white", linewidths=1.5)
        ax.set_title(f"Head {head_idx}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(num_heads, 2 * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heads comparison to {save_path}")

    return fig


def plot_attention_pattern_summary(
    attention_weights_list: List[torch.Tensor],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Create summary visualization showing attention patterns across layers.
    
    Args:
        attention_weights_list: List of attention tensors, one per layer.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
    """
    n_layers = len(attention_weights_list)

    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=figsize)
    axes = axes.flatten()

    fig.suptitle("Attention Patterns Across Layers", fontsize=14, fontweight="bold")

    for layer_idx, attn in enumerate(attention_weights_list):
        ax = axes[layer_idx]

        # Average over batch and heads
        attn_avg = attn.mean(dim=(0, 1)).detach().cpu().numpy()  # (N, N)

        im = ax.imshow(attn_avg, cmap="viridis", aspect="auto")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention summary to {save_path}")

    return fig


if __name__ == "__main__":
    print("Attention Visualization — Demo")
    print("=" * 40)

    # Create synthetic attention weights for demo
    B, num_heads, N = 1, 8, 128
    n_patches_h, n_patches_w = 8, 16

    # Create attention that's slightly focused (not uniform)
    attn = torch.rand(B, num_heads, N, N)
    attn = attn.softmax(dim=-1)

    fig = plot_attention_maps(
        attn,
        layer_idx=0,
        head_idx=0,
        n_patches_h=n_patches_h,
        n_patches_w=n_patches_w,
        save_path="results/figures/attention_maps.png",
    )

    print("✅ Attention visualization created!")
    plt.show()
