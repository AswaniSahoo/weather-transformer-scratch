"""
Unit tests for Trainer configuration handling.

Run with: pytest tests/test_trainer.py -v
"""

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.weather_transformer import WeatherTransformer
from src.training.trainer import Trainer

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def tiny_model() -> WeatherTransformer:
    return WeatherTransformer(
        in_channels=4, out_channels=4, img_height=8, img_width=16,
        patch_size=4, embed_dim=16, num_heads=2, num_layers=1,
    )


def tiny_loader() -> DataLoader:
    x = torch.randn(8, 4, 8, 16)
    return DataLoader(TensorDataset(x, x + 0.1 * torch.randn_like(x)), batch_size=4)


@pytest.fixture
def default_config(tmp_path):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    return config


class TestTrainerConfig:
    """The trainer must use the values written in the YAML config."""

    def test_reads_top_level_loss_section(self, default_config):
        default_config["loss"] = {
            "mse_weight": 2.0, "smoothness_weight": 0.3, "conservation_weight": 0.7,
        }
        trainer = Trainer(tiny_model(), tiny_loader(), tiny_loader(), default_config, "cpu")

        assert trainer.loss_fn.mse_weight == 2.0
        assert trainer.loss_fn.smoothness_weight == 0.3
        assert trainer.loss_fn.conservation_weight == 0.7

    def test_reads_training_section(self, default_config, tmp_path):
        default_config["training"].update(
            {"gradient_clip": 0.5, "min_lr": 1e-5, "save_every": 3}
        )
        trainer = Trainer(tiny_model(), tiny_loader(), tiny_loader(), default_config, "cpu")

        assert trainer.gradient_clip == 0.5
        assert trainer.scheduler.eta_min == 1e-5
        assert trainer.save_every == 3
        assert trainer.checkpoint_dir == tmp_path / "ckpt"

    def test_short_run_writes_checkpoints_and_log(self, default_config, tmp_path):
        default_config["training"].update({"epochs": 2, "save_every": 1})
        trainer = Trainer(tiny_model(), tiny_loader(), tiny_loader(), default_config, "cpu")

        history = trainer.train()

        ckpt_dir = tmp_path / "ckpt"
        assert (ckpt_dir / "best_model.pt").exists()
        assert (ckpt_dir / "checkpoint_epoch_1.pt").exists()
        assert (ckpt_dir / "checkpoint_epoch_2.pt").exists()
        log = json.loads((ckpt_dir / "training_log.json").read_text())
        assert len(log["history"]["train_loss"]) == len(history["train_loss"]) == 2
