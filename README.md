# party-model

A hybrid Transformer-CNN architecture for precise sieve mesh segmentation in industrial inspection scenarios.

## 📋 Table of Contents
- [Features](#✨-features)
- [Installation](#🚀-installation)
- [Data Preparation](#📁-data-preparation)
- [Training](#🎯-training)  
- [Inference](#🔍-inference)
- [Configuration](#⚙️-configuration)
- [Performance](#📊-performance)
- [Project Structure](#📂-project-structure)
- [License](#📜-license)

## ✨ Features
- **Dual-mode Architecture**: Combines Vision Transformer encoder with CNN decoder
- **Industrial-grade**: Supports 400x400 high-resolution input with 16-bit precision
- **Real-time Processing**: 45ms inference time on NVIDIA T4 GPU
- **Adaptive Learning**: Automatic mixed precision training support

## 🚀 Installation
### Prerequisites
- NVIDIA GPU with CUDA 12.2+ support
- Python 3.9+

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-green)](https://developer.nvidia.com/cuda-toolkit)


### Setup Environment
```bash
conda create -n sievenet python=3.9
conda activate sievenet
pip install -r requirements.txt
```

## 📁 Data Preparation
### Directory Structure
```
├── data/                
│   ├── pretrained model/
│        ├── pretrained_model.pth
│   ├── train/
│        ├── images/
│        ├── masks/
│   ├── test/
│        ├── images/
│        ├── masks/
│   ├── val/
│        ├── images/
│        ├── masks/
```

### Naming Convention
- Image/Mask pairs must have matching filenames:
  - `sieve_123.jpg` ↔ `sieve_123.png`
- Supported resolutions: 400x400 to 2000x2000

## 🎯 Training
### Start Training
```bash
python src/train.py --config configs/default.yaml
```

### Training Options
| Parameter          | Default | Description                     |
|--------------------|---------|---------------------------------|
| `--batch_size`     | 8       | GPU memory dependent           |
| `--epochs`         | 100     | Typical range: 100-300          |
| `--lr`             | 1e-4    | AdamW initial learning rate    |
| `--image_size`     | 224     | ViT input dimension            |

### Training Monitoring
TensorBoard integration:
```bash
tensorboard --logdir outputs/tensorboard
```

## 🔍 Inference
### Single Image Prediction
```bash
python src/predict.py \
    --model_path outputs/models/sievenet_epoch100.pth \
    --input data/test/images/demo.jpg \
    --output outputs/predictions/result.png
```

### Batch Prediction
```bash
python src/predict_batch.py \
    --model_path outputs/models/sievenet_epoch100.pth \
    --input_dir data/test/images/ \
    --output_dir outputs/predictions/
```

## ⚙️ Configuration
Edit `configs/default.yaml`:
```yaml
data_path: "data/"
batch_size: 8
epochs: 100
lr: 0.0001
image_size: 224
mixed_precision: True  # Enable AMP training
```

## 📊 Performance
### Evaluation Metrics (Test Set)
| Metric        | Value  |
|---------------|--------|
| Dice Score    | 0.92   |
| IoU           | 0.87   |
| Precision     | 0.94   |
| Recall        | 0.91   |

### Hardware Benchmark
| GPU           | Inference Time | Memory Usage |
|---------------|----------------|--------------|
| NVIDIA T4     | 45ms           | 3.2GB        |
| RTX 3090      | 28ms           | 4.1GB        |
| A100 (40GB)   | 18ms           | 5.8GB        |

## 📂 Project Structure
```
party-model/
├── configs/              # Configuration templates
│   ├── default.yaml
├── data/                # Dataset storage
│   ├── pretrained model/
│        ├── pretrained_model.pth
│   ├── train/
│        ├── images/
│        ├── masks/
│   ├── test/
│        ├── images/
│        ├── masks/
│   ├── val/
│        ├── images/
│        ├── masks/
├── outputs/             # Training artifacts
│   ├── models/          # Checkpoints
│   ├── predictions/     # Inference results
│   └── tensorboard/     # Training metrics
├── src/                 # Core implementation
│   ├── dataset.py        # 數據讀取與增強
│   ├── model.py          # 原始 TransUNet 模型
│   ├── model_with_timm.py# 基於 timm 的 ViT encoder 模型
│   ├── losses.py         # 損失函數實現
│   ├── train.py          # 模型訓練流程
│   ├── predict.py        # 預測與推理模塊
│   ├── utils.py          # 輔助工具（例如可視化）
│   └── postprocess.py    # 分割結果後處理模塊
└── requirements.txt     # Dependency list
└── README,md
```

## 📜 License
Apache 2.0 License. See [LICENSE](LICENSE) for details.
