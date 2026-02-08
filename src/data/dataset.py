"""
PyTorch Dataset for WeatherBench2 ERA5 data.

Loads preprocessed .npy files and serves (input, target) pairs
for training the Weather Transformer.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WeatherBenchDataset(Dataset):
    """
    PyTorch Dataset for WeatherBench2 weather forecasting.
    
    Loads preprocessed numpy arrays and returns (input, target) tensor pairs.
    Input: weather state at time t, shape (C, H, W)
    Target: weather state at time t+1, shape (C, H, W)
    
    Args:
        data_dir: Path to processed data directory
        split: One of "train", "val", "test"
        transform: Optional transform to apply to both input and target
    """
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load preprocessed data
        inputs_path = self.data_dir / f"{split}_inputs.npy"
        targets_path = self.data_dir / f"{split}_targets.npy"
        
        if not inputs_path.exists():
            raise FileNotFoundError(
                f"Data not found at {inputs_path}. "
                f"Run preprocessing first: python -m src.data.preprocessing"
            )
        
        self.inputs = np.load(inputs_path)
        self.targets = np.load(targets_path)
        
        # Validate shapes match
        assert self.inputs.shape == self.targets.shape, (
            f"Input/target shape mismatch: {self.inputs.shape} vs {self.targets.shape}"
        )
        
        self.n_samples = self.inputs.shape[0]
        self.n_channels = self.inputs.shape[1]
        self.height = self.inputs.shape[2]
        self.width = self.inputs.shape[3]
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input, target) pair.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_tensor, target_tensor), both shape (C, H, W)
        """
        input_arr = self.inputs[idx]
        target_arr = self.targets[idx]
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_arr).float()
        target_tensor = torch.from_numpy(target_arr).float()
        
        # Apply transform if provided
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor, target_tensor
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return (N, C, H, W) shape of the dataset."""
        return (self.n_samples, self.n_channels, self.height, self.width)


def create_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = WeatherBenchDataset(data_dir, split="train")
    val_dataset = WeatherBenchDataset(data_dir, split="val")
    test_dataset = WeatherBenchDataset(data_dir, split="test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader
