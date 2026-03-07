# 🌦️ Weather Transformer from Scratch

> A physics-aware Vision Transformer for weather forecasting, built entirely from scratch in PyTorch.


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-74%20passed-brightgreen.svg)]()

---

## 🎯 Project Overview

This project implements a **Vision Transformer (ViT)** architecture for weather forecasting, trained on ERA5 reanalysis data from WeatherBench2. Key features:

- ✅ **Every component built from scratch** — no `nn.MultiheadAttention`
- ✅ **Physics-informed loss** — MSE + spatial smoothness + conservation constraints
- ✅ **Real climate data** — WeatherBench2 / ERA5 at 5.625° resolution (2015–2020)
- ✅ **Comprehensive evaluation** — RMSE, MAE, ACC vs persistence baseline
- ✅ **Production-ready** — Config-driven training, checkpointing, logging
- ✅ **74 unit tests** — Full test coverage across data, model, and metrics

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Weather Transformer                      │
│                    4,805,440 parameters                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input (B, 4, 32, 64)     4 weather variables, lat×lon     │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────────┐                                      │
│   │  Patch Embedding │  Conv2d → (B, 128, 256)              │
│   │  (4×4 patches)   │  128 patches, 256-dim embeddings     │
│   └────────┬─────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌──────────────────┐                                      │
│   │  + Positional    │  Learnable spatial position info     │
│   │    Encoding      │                                      │
│   └────────┬─────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌──────────────────┐                                      │
│   │  Transformer     │  × 6 layers                          │
│   │  Encoder Blocks  │  - Multi-Head Self-Attention (8h)    │
│   │                  │  - Feed-Forward MLP (4× expansion)   │
│   │                  │  - Pre-Norm + Residual connections   │
│   └────────┬─────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌──────────────────┐                                      │
│   │  Prediction Head │  Linear → Reshape → (B, 4, 32, 64)   │
│   └──────────────────┘                                      │
│                                                             │
│   Output: Predicted weather state at t+6h                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Results

### Model vs Persistence Baseline (2020 Test Set)

| Metric | Model | Persistence | Improvement |
|--------|-------|-------------|-------------|
| **RMSE** | 0.197 | 0.270 | **27.0% ✅** |
| **MAE** | 0.126 | 0.147 | **13.9% ✅** |
| **ACC** | 0.955 | 0.912 | **+4.3 pts ✅** |

> *Evaluated on 1,316 test samples from year 2020. Persistence baseline: predict Y(t+1) = X(t).*

### Per-Variable RMSE

| Variable | RMSE | Description |
|----------|------|-------------|
| t850 | **0.083** | Temperature at 850 hPa — *best predicted* |
| z500 | **0.067** | Geopotential at 500 hPa — *smoothest field* |
| u10 | **0.242** | U-wind at 10m — *more chaotic* |
| v10 | **0.292** | V-wind at 10m — *hardest to predict* |

> Wind components (u10, v10) have higher RMSE because wind fields are inherently more turbulent and less spatially smooth than temperature and geopotential fields.

### Training Details

| Property | Value |
|----------|-------|
| Parameters | 4,805,440 |
| Training samples | 6,136 (2015–2018) |
| Validation samples | 1,315 (2019) |
| Test samples | 1,316 (2020) |
| Epochs | 50 |
| Training time | ~12 min (NVIDIA GPU) |
| Best epoch | 49 |
| Best val loss | 0.0699 |
| Convergence | Smooth = every epoch improved ⭐ |

### Sample Prediction

![Prediction Sample](results/figures/prediction_sample.png)

*6-hour forecast for temperature at 850 hPa. Left: Input, Middle: Prediction, Right: Ground Truth*

---

## 📁 Project Structure

```
weather-transformer-scratch/
├── configs/
│   └── default.yaml              # Hyperparameters & paths
├── src/
│   ├── data/
│   │   ├── download.py           # WeatherBench2 data download
│   │   ├── preprocessing.py      # Preprocessing & normalization
│   │   └── dataset.py            # PyTorch Dataset & DataLoader
│   ├── models/
│   │   ├── patch_embedding.py    # Image → patch tokens
│   │   ├── positional_encoding.py # Learnable + Sinusoidal PE
│   │   ├── attention.py          # Multi-head self-attention (from scratch)
│   │   ├── transformer_block.py  # Attention + MLP + residual + pre-norm
│   │   ├── weather_transformer.py # Full model assembly
│   │   └── physics_loss.py       # MSE + smoothness + conservation loss
│   ├── training/
│   │   └── trainer.py            # Training loop with checkpointing
│   ├── evaluation/
│   │   ├── metrics.py            # RMSE, MAE, ACC, persistence baseline
│   │   └── evaluate.py           # Evaluation script
│   └── visualization/
│       ├── plot_predictions.py   # World map predictions with cartopy
│       ├── plot_loss.py          # Training/validation loss curves
│       └── plot_attention.py     # Attention weight visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb # Data inspection & statistics
│   ├── 02_model_walkthrough.ipynb # Step-by-step model building
│   └── 03_results_analysis.ipynb # Results & visualizations
├── scripts/
│   └── train.py                  # Training entry point
├── tests/
│   ├── test_dataset.py           # Dataset tests (10)
│   ├── test_model.py             # Model tests (51)
│   └── test_metrics.py           # Metrics tests (13)
├── checkpoints/                  # Saved model weights
├── results/                      # Evaluation outputs & figures
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AswaniSahoo/weather-transformer-scratch.git
cd weather-transformer-scratch

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For GPU training (NVIDIA GPUs), install PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download Data

```bash
python src/data/download.py --start-year 2015 --end-year 2020
python src/data/preprocessing.py
```

### 3. Train Model

```bash
# CPU training
python scripts/train.py --config configs/default.yaml

# GPU training (recommended)
python scripts/train.py --config configs/default.yaml --device cuda

# GPU training with custom epochs
python scripts/train.py --config configs/default.yaml --device cuda --epochs 50
```

### 4. Evaluate

```bash
python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 🔧 Configuration

All hyperparameters are defined in `configs/default.yaml`:

```yaml
model:
  in_channels: 4
  embed_dim: 256
  num_heads: 8
  num_layers: 6
  patch_size: 4
  mlp_ratio: 4.0
  dropout: 0.1

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 0.01
  loss:
    mse_weight: 1.0
    smoothness_weight: 0.1
    conservation_weight: 0.05

data:
  variables: [t850, z500, u10, v10]
  lat_size: 32
  lon_size: 64
```

---

## 📈 Weather Variables

| Variable | Description | Level | Unit |
|----------|-------------|-------|------|
| `t850` | Temperature | 850 hPa | K |
| `z500` | Geopotential | 500 hPa | m²/s² |
| `u10` | U-wind component | 10m | m/s |
| `v10` | V-wind component | 10m | m/s |

---

## 🧪 Testing

The project includes **74 comprehensive unit tests** across 3 test files:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_model.py` | 51 | Patch embedding, positional encoding, attention, transformer block, full model, physics loss |
| `test_dataset.py` | 10 | Data loading, tensor shapes, normalization, DataLoader |
| `test_metrics.py` | 13 | RMSE, MAE, ACC, persistence baseline |

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Key Components

### Physics-Informed Loss

```python
L = α × MSE + β × Smoothness + γ × Conservation

# MSE: Standard pixel-wise reconstruction
# Smoothness: Penalizes unrealistic spatial gradients (∂T/∂x, ∂T/∂y)
# Conservation: Predicted global mean ≈ target global mean (energy proxy)
```

### Multi-Head Self-Attention (from scratch)

```python
# No nn.MultiheadAttention — built entirely manually!
Q = x @ W_q  # Query projection
K = x @ W_k  # Key projection
V = x @ W_v  # Value projection

attention = softmax(Q @ K.T / sqrt(d_k)) @ V
```

---

## 📖 References

1. **GraphCast** — Lam et al., DeepMind (2023) — [arXiv:2212.12794](https://arxiv.org/abs/2212.12794)
2. **FourCastNet** — Pathak et al., NVIDIA (2022) — [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)
3. **ClimaX** — Nguyen et al., Microsoft (2023) — [arXiv:2301.10343](https://arxiv.org/abs/2301.10343)
4. **WeatherBench2** — Rasp et al. (2023) — [arXiv:2308.15560](https://arxiv.org/abs/2308.15560)
5. **Attention Is All You Need** — Vaswani et al. (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
6. **Vision Transformer (ViT)** — Dosovitskiy et al. (2020) — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## 👤 Author

**Aswani Sahoo**

- GitHub: [@AswaniSahoo](https://github.com/AswaniSahoo)
- LinkedIn: [Aswani Sahoo](https://linkedin.com/in/aswanisahoo)

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
