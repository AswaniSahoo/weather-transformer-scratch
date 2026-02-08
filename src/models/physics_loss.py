"""
Physics-Informed Loss Functions for weather forecasting.

Goes beyond standard MSE by adding physical constraints:
1. MSE Loss — standard pixel-wise reconstruction
2. Spatial Smoothness — penalizes unrealistic sharp gradients
3. Conservation Loss — predicted global mean ≈ target global mean (energy proxy)

Combined: L = α * MSE + β * Smoothness + γ * Conservation
"""

import torch
import torch.nn as nn


class SpatialSmoothnessLoss(nn.Module):
    """
    Penalizes large spatial gradients in predictions.
    
    Weather fields should be spatially smooth — sharp discontinuities
    between adjacent grid cells are physically unrealistic.
    
    Computes finite differences along latitude and longitude axes
    and penalizes their magnitude.
    """

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial smoothness penalty.
        
        Args:
            prediction: Predicted weather grid (B, C, H, W).
            
        Returns:
            Scalar smoothness loss.
        """
        # Finite differences along latitude (height) axis
        # ∂T/∂y ≈ T[..., h+1, :] - T[..., h, :]
        diff_lat = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]

        # Finite differences along longitude (width) axis
        # ∂T/∂x ≈ T[..., :, w+1] - T[..., :, w]
        diff_lon = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]

        # Mean squared gradient magnitude
        smoothness = diff_lat.pow(2).mean() + diff_lon.pow(2).mean()

        return smoothness


class ConservationLoss(nn.Module):
    """
    Penalizes violation of global conservation (energy proxy).
    
    The global mean of weather fields should not change drastically
    between timesteps. This acts as a soft constraint on conservation
    of energy/mass.
    
    L_conservation = MSE(mean(prediction), mean(target))
    
    Computed per-variable then averaged.
    """

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute conservation loss.
        
        Args:
            prediction: Predicted weather grid (B, C, H, W).
            target: Target weather grid (B, C, H, W).
            
        Returns:
            Scalar conservation loss.
        """
        # Global spatial mean per sample per variable: (B, C)
        pred_mean = prediction.mean(dim=(-2, -1))
        target_mean = target.mean(dim=(-2, -1))

        # MSE between global means
        conservation = (pred_mean - target_mean).pow(2).mean()

        return conservation


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss function.
    
    L = α * MSE + β * Smoothness + γ * Conservation
    
    Args:
        mse_weight: Weight for standard MSE loss (α).
        smoothness_weight: Weight for spatial smoothness penalty (β).
        conservation_weight: Weight for conservation constraint (γ).
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        conservation_weight: float = 0.05,
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.conservation_weight = conservation_weight

        self.mse_loss = nn.MSELoss()
        self.smoothness_loss = SpatialSmoothnessLoss()
        self.conservation_loss = ConservationLoss()

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> dict:
        """
        Compute combined physics-informed loss.
        
        Args:
            prediction: Predicted weather grid (B, C, H, W).
            target: Target weather grid (B, C, H, W).
            
        Returns:
            Dict with keys:
                "total": Combined weighted loss (for backward).
                "mse": MSE component value.
                "smoothness": Smoothness component value.
                "conservation": Conservation component value.
        """
        # Individual components
        mse = self.mse_loss(prediction, target)
        smoothness = self.smoothness_loss(prediction)
        conservation = self.conservation_loss(prediction, target)

        # Weighted combination
        total = (
            self.mse_weight * mse
            + self.smoothness_weight * smoothness
            + self.conservation_weight * conservation
        )

        return {
            "total": total,
            "mse": mse.detach(),
            "smoothness": smoothness.detach(),
            "conservation": conservation.detach(),
        }


if __name__ == "__main__":
    print("Physics-Informed Loss — Sanity Check")
    print("=" * 45)

    B, C, H, W = 4, 4, 32, 64

    prediction = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randn(B, C, H, W)

    loss_fn = PhysicsInformedLoss(
        mse_weight=1.0, smoothness_weight=0.1, conservation_weight=0.05
    )

    losses = loss_fn(prediction, target)

    print(f"Total loss:        {losses['total'].item():.4f}")
    print(f"MSE component:     {losses['mse'].item():.4f}")
    print(f"Smoothness:        {losses['smoothness'].item():.4f}")
    print(f"Conservation:      {losses['conservation'].item():.4f}")

    # Test backward
    losses["total"].backward()
    print(f"Gradient exists:   {prediction.grad is not None}")
    print(f"Gradient shape:    {prediction.grad.shape}")

    print(f"\n✅ Physics-informed loss working!")
