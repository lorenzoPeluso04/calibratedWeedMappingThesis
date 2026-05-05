# 🌱 Calibrated Weed Mapping

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-ECAI%202025-red.svg)](https://ecai2025.org/)
[![Workshop](https://img.shields.io/badge/workshop-Green--aware%20AI%202025-brightgreen.svg)](https://green-aware2025.web.app/)

*Precision agriculture meets reliable AI through calibrated confidence estimation*

</div>

---

## 📖 Overview

This repository contains the official implementation of **"Calibrated Weed Mapping"**, accepted at the **2nd Workshop on Green-Aware Artificial Intelligence** co-located with the **28th European Conference on Artificial Intelligence (ECAI 2025)** in **Bologna, Italy**.

Our work addresses the critical challenge of reliable weed detection in precision agriculture by introducing calibrated confidence estimation techniques that ensure AI predictions are not only accurate but also trustworthy. This is essential for real-world deployment where overconfident or underconfident predictions can lead to suboptimal agricultural decisions.

### ✨ Key Features

- 🎯 **State-of-the-art weed detection** using modern architectures (MobileNetV4, etc.)
- 🔧 **Calibration techniques** for reliable confidence estimation
- 🌾 **Agricultural focus** with real-world applicability
- 📊 **Comprehensive evaluation** metrics and visualizations
- 🚀 **Easy-to-use** command-line interface

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

Install all required dependencies using uv:

```bash
uv sync
```

---

## 📊 Dataset Setup

This repository builds upon the preprocessing pipeline from [RoWeeder](https://github.com/yourusername/RoWeeder). Choose one of the following options:

### Option 1: Download Preprocessed Dataset (Recommended) 🌟

```bash
bash download.sh
```

### Option 2: Manual Dataset Preparation

If you prefer to preprocess the data yourself:

1. **Clone RoWeeder repository**
2. **Download and extract the dataset:**

```bash
wget http://robotics.ethz.ch/~asl-datasets/2018-weedMap-dataset-release/Orthomosaic/RedEdge.zip
unzip RedEdge.zip -d RoWeeder/dataset
```

3. **Apply rotations to orthomosaics:**

```bash
for i in {000..004}; do
    python3 RoWeeder/main.py rotate \
        --root RoWeeder/dataset/RedEdge/$i \
        --outdir RoWeeder/dataset/rotated_ortho/$i \
        --angle -46
done
```

4. **Generate patches:**

```bash
for i in {000..004}; do
    python3 RoWeeder/main.py patchify \
        --root RoWeeder/dataset/rotated_ortho/$i \
        --outdir RoWeeder/dataset/patches/512/$i \
        --patch_size 512
done
```

---

## 🧪 Experiments

All experiments can be reproduced using the commands in `scripts.sh`. The workflow consists of three main phases:

### 1. Training 🏋️‍♂️

Train your model with various architectures and loss functions:

```bash
# Example: MobileNetV4 with Focal Loss
python main.py train --model mobilenetv4 --loss focal --gamma 2.0
```

Training checkpoints will be saved to the `weights/` directory.

### 2. Calibration 🎯

Apply calibration techniques to improve confidence estimation:

```bash
python main.py calibrate \
    --model mobilenetv4 \
    --calibration_technique temperature_scaling \
    --num_epochs 30 \
    --checkpoint weights/mobilenetv4_focal_gamma2.0.pth
```

### 3. Evaluation 📈

Test your models with or without calibration:

**Original model:**
```bash
python main.py test \
    --model mobilenetv4 \
    --checkpoint weights/mobilenetv4_focal_gamma2.0.pth
```

**Calibrated model:**
```bash
python main.py evaluate \
    --model mobilenetv4 \
    --calibration_technique temperature_scaling \
    --calibration_params weights/mobilenetv4_calibrated_n30_temperature_scaling_ckpt_mobilenetv4_focal_gamma2.pkl \
    --checkpoint weights/mobilenetv4_focal_gamma2.0.pth
```

---

## 📁 Repository Structure

```
📦 CalibratedWeedMapping
├─ 📄 .gitignore
├─ 🐍 .python-version
├─ 🔧 .vscode/
│  └─ launch.json
├─ 📖 README.md
├─ 📊 calibration.ipynb          # Calibration analysis notebook
├─ 🔬 calweed/                   # Core package
│  ├─ calibrate.py               # Calibration methods
│  ├─ data.py                    # Data loading utilities
│  ├─ evaluate.py                # Evaluation scripts
│  ├─ metrics.py                 # Performance metrics
│  ├─ model.py                   # Model architectures
│  ├─ train.py                   # Training pipeline
│  └─ weedmap.py                 # Main weed mapping logic
├─ ⬇️  download.sh                # Dataset download script
├─ 🚀 main.py                    # CLI entry point
├─ 📦 pyproject.toml             # Project configuration
├─ 📈 qualitative.ipynb          # Qualitative results analysis
├─ 🔧 script.sh                  # Experiment scripts
└─ 🔒 uv.lock                    # Dependency lock file
```

---

## 📚 Citation

If you use this work in your research, please cite our paper:

```bibtex
@inproceedings{demarinis2025calibrated,
    title={Calibrated Weed Mapping},
    author={De Marinis, Pasquale and Detomaso, Gabriele and Vessio, Gennaro and Castellano, Giovanna},
    booktitle={Proceedings of the 2nd Workshop on Green-Aware Artificial Intelligence (Green-Aware AI 2025) co-located with the 28th European Conference on Artificial Intelligence (ECAI 2025)},
    year={2025},
    address={Bologna, Italy},
    series={CEUR Workshop Proceedings},
    publisher={CEUR-WS.org},
    volume={TBD},
    url={https://ceur-ws.org/},
    note={Workshop proceedings to be published}
}
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🌱 Advancing sustainable agriculture through reliable AI 🌱**

</div>