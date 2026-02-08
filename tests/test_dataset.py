"""
Unit tests for WeatherBenchDataset.

Tests data loading, tensor shapes, normalization ranges, and DataLoader.
Run with: pytest tests/test_dataset.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import WeatherBenchDataset, create_dataloaders


# Test constants matching expected data shape
N_SAMPLES = 100
N_CHANNELS = 4  # t850, z500, u10, v10
HEIGHT = 32
WIDTH = 64


@pytest.fixture
def synthetic_data_dir():
    """Create temporary directory with synthetic test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic normalized data (mean ~0, std ~1)
        for split in ["train", "val", "test"]:
            n = N_SAMPLES if split == "train" else N_SAMPLES // 4
            
            inputs = np.random.randn(n, N_CHANNELS, HEIGHT, WIDTH).astype(np.float32)
            targets = np.random.randn(n, N_CHANNELS, HEIGHT, WIDTH).astype(np.float32)
            
            np.save(tmpdir / f"{split}_inputs.npy", inputs)
            np.save(tmpdir / f"{split}_targets.npy", targets)
        
        yield str(tmpdir)


class TestWeatherBenchDataset:
    """Tests for WeatherBenchDataset class."""
    
    def test_dataset_length(self, synthetic_data_dir):
        """Test that dataset length matches number of samples."""
        dataset = WeatherBenchDataset(synthetic_data_dir, split="train")
        assert len(dataset) == N_SAMPLES
    
    def test_dataset_getitem_shape(self, synthetic_data_dir):
        """Test that __getitem__ returns correct shapes."""
        dataset = WeatherBenchDataset(synthetic_data_dir, split="train")
        input_tensor, target_tensor = dataset[0]
        
        assert input_tensor.shape == (N_CHANNELS, HEIGHT, WIDTH)
        assert target_tensor.shape == (N_CHANNELS, HEIGHT, WIDTH)
    
    def test_dataset_tensor_type(self, synthetic_data_dir):
        """Test that returned tensors are float32."""
        dataset = WeatherBenchDataset(synthetic_data_dir, split="train")
        input_tensor, target_tensor = dataset[0]
        
        assert input_tensor.dtype == torch.float32
        assert target_tensor.dtype == torch.float32
    
    def test_dataset_shape_property(self, synthetic_data_dir):
        """Test the shape property returns correct (N, C, H, W)."""
        dataset = WeatherBenchDataset(synthetic_data_dir, split="train")
        assert dataset.shape == (N_SAMPLES, N_CHANNELS, HEIGHT, WIDTH)
    
    def test_normalization_range(self, synthetic_data_dir):
        """Test that normalized data is roughly in standard normal range."""
        dataset = WeatherBenchDataset(synthetic_data_dir, split="train")
        
        # Check a few samples
        for i in range(min(10, len(dataset))):
            input_tensor, _ = dataset[i]
            
            # For normalized data, most values should be within [-3, 3]
            assert input_tensor.min() > -10, "Data appears unnormalized (too low)"
            assert input_tensor.max() < 10, "Data appears unnormalized (too high)"
    
    def test_different_splits(self, synthetic_data_dir):
        """Test that different splits load correctly."""
        train_ds = WeatherBenchDataset(synthetic_data_dir, split="train")
        val_ds = WeatherBenchDataset(synthetic_data_dir, split="val")
        test_ds = WeatherBenchDataset(synthetic_data_dir, split="test")
        
        # Train should be larger
        assert len(train_ds) == N_SAMPLES
        assert len(val_ds) == N_SAMPLES // 4
        assert len(test_ds) == N_SAMPLES // 4
    
    def test_missing_data_raises_error(self):
        """Test that missing data file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            WeatherBenchDataset("/nonexistent/path", split="train")


class TestDataLoaders:
    """Tests for create_dataloaders function."""
    
    def test_create_dataloaders(self, synthetic_data_dir):
        """Test that create_dataloaders returns three DataLoaders."""
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=synthetic_data_dir,
            batch_size=16,
            num_workers=0,
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
    
    def test_dataloader_batch_shape(self, synthetic_data_dir):
        """Test that DataLoader yields correct batch shapes."""
        train_loader, _, _ = create_dataloaders(
            data_dir=synthetic_data_dir,
            batch_size=16,
            num_workers=0,
        )
        
        batch_input, batch_target = next(iter(train_loader))
        
        assert batch_input.shape == (16, N_CHANNELS, HEIGHT, WIDTH)
        assert batch_target.shape == (16, N_CHANNELS, HEIGHT, WIDTH)
    
    def test_dataloader_iteration(self, synthetic_data_dir):
        """Test that we can iterate through the DataLoader."""
        train_loader, _, _ = create_dataloaders(
            data_dir=synthetic_data_dir,
            batch_size=32,
            num_workers=0,
        )
        
        total_samples = 0
        for batch_input, batch_target in train_loader:
            total_samples += batch_input.shape[0]
        
        assert total_samples == N_SAMPLES
