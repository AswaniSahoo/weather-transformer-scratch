# 🌦️ Weather Transformer from Scratch

> A physics-aware Vision Transformer for weather forecasting, built entirely from scratch in PyTorch.

## 🚧 Work in Progress

This project implements a Vision Transformer (ViT) architecture to predict next-step weather states (temperature, geopotential, wind) from gridded ERA5 reanalysis data — with physics-informed loss functions.

## 🎯 Project Goals

- Build every transformer component from scratch (no `nn.MultiheadAttention`)
- Train on real climate data (WeatherBench2 / ERA5)
- Add physics-informed constraints (smoothness, conservation)
- Beat the persistence baseline on standard weather metrics
- Visualize predictions on world maps with proper projections

## 🏗️ Architecture

```
Input (B, 4, 32, 64)     — 4 weather variables on a lat/lon grid
       ↓
Patch Embedding           — Split into spatial patches, project to embeddings
       ↓
+ Positional Encoding     — Learnable spatial position embeddings
       ↓
N × Transformer Blocks    — Multi-head self-attention + MLP + residual
       ↓
Prediction Head           — Linear projection back to grid space
       ↓
Output (B, 4, 32, 64)    — Predicted weather state at t+6h
```

## 📊 Variables

| Variable | Description | Level |
|----------|-------------|-------|
| `t850`   | Temperature | 850 hPa |
| `z500`   | Geopotential | 500 hPa |
| `u10`    | U-component of wind | 10m |
| `v10`    | V-component of wind | 10m |

## 🛠️ Tech Stack

- **PyTorch** — Model & training
- **xarray** — Climate data handling
- **Cartopy** — Map visualizations
- **WeatherBench2** — Benchmark dataset

## 📁 Project Structure

```
weather-transformer-scratch/
├── configs/              # Hyperparameters & paths
├── src/
│   ├── data/             # Data download, dataset, preprocessing
│   ├── models/           # All model components from scratch
│   ├── training/         # Training loop & scheduler
│   ├── evaluation/       # Metrics & evaluation scripts
│   └── utils/            # Utilities
├── notebooks/            # Exploration & analysis notebooks
├── tests/                # Unit tests
├── checkpoints/          # Saved model weights
└── logs/                 # Training logs
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python src/data/download.py

# Train model
python scripts/train.py --config configs/default.yaml

# Evaluate
python scripts/predict.py --config configs/default.yaml
```

## 📚 References

- [GraphCast — DeepMind (2023)](https://arxiv.org/abs/2212.12794)
- [FourCastNet — NVIDIA (2022)](https://arxiv.org/abs/2202.11214)
- [ClimaX — Microsoft (2023)](https://arxiv.org/abs/2301.10343)
- [WeatherBench2 Benchmark](https://arxiv.org/abs/2308.15560)

## 📝 License

MIT License — see [LICENSE](LICENSE)

---

*Built as preparation for GSoC 2026 — AI for Science*
