# Fusion of Skeletal and Racket Dynamics for Enhanced Table Tennis Action Recognition (SRF-Net)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper **"Fusion of Skeletal and Racket Dynamics for Enhanced Table Tennis Action Recognition"**, currently under review at *The Visual Computer*.

## 📄 Introduction

We propose **SRF-Net** (Skeleton-Racket Fusion Network), a multimodal framework that integrates human skeletal keypoints and racket motion trajectories to analyze table tennis actions.

**Key Features:**

* **Multimodal Fusion:** Combines **MediaPipe** (Body Pose) and **YOLOv8** (Racket Detection) streams.
* **AST-GCN-TA:** A novel adaptive graph convolutional network with temporal attention to capture non-local joint dependencies and critical hitting frames.

**Dataset:**
The dataset covers **10 action categories** (e.g., Forehand Drive, Backhand Loop, Banana Flick) collected from university-level players.

## 📂 Repository Structure

```text
SRF-Net/
├── data/                    # Dataset folder (Place your data here)
│   ├── racket_csv/          # Processed skeleton files (.csv)
│   ├── skeleton_npy/        # Processed racket trajectory files (.npy)
│   └── raw_videos/          # Original MP4 videos (Samples)
├── racket_detection/        # YOLOv8 scripts for racket tracking
│   ├── runs                 # YOLOv8 training logs and weights
│   ├── yolo_data            # Dataset for YOLOv8 training (images/labels)
│   ├── train_yolo.py        # Training script for YOLO
│   ├── predict.py           # Inference script for tracking
│   └── extract_racket.py    # Extract racket to csv (Using Yolov8)
├── models.py                # Definition of AST-GCN-TA and baselines (LSTM, ST-GCN, etc.)
├── train_fusion.py          # Main training script for SRF-Net (Fusion)
├── train.py                 # Training script for single(racket/skeletion) 
├── extract_skeleton.py      # Extract skeletal to npy (Using Mediapipe)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🛠️ Prerequisites

The code is implemented in Python 3.9 using PyTorch 1.13.1.

```bash
pip install -r requirements.txt
```

## 🚀 Usage Guidelines

### Step 1: Data Preprocessing (Feature Extraction)

#### Extract Skeletal Keypoints: Use MediaPipe to extract upper-body keypoints from videos in data/raw_videos

```bash
python extract_skeleton.py --input data/raw_videos --output data/skeleton_npy
```

#### Extract Racket Trajectories: Use the trained YOLOv8 model to generate racket trajectory files

```bash
cd racket_detection
# Ensure your trained weights are in runs/ or specified path
python extract_racket.py 
cd ..
```

### Step 2: Training SRF-Net (Fusion Model)

#### To train the proposed multi-feature fusion model

```bash
# Train on Fusion data (Skeleton and Racket)
python train_fusion.py --model STGCN_AG_TA --epochs 100
```

### Step 3:Training Baselines (Single Modality)

#### To train single-stream models (Skeleton-only or Racket-only) for comparison

```bash
# Train on Skeleton data
python train.py --type skeleton --model STGCN

# Train on Racket data
python train.py --type racket --model LSTM
```

### Optional: Training Racket Detector (YOLOv8)

If you wish to retrain the racket detector from scratch using your own data:
Prepare your dataset in racket_detection/yolo_data/ with a valid .yaml config.
Run the training script:

```bash
cd racket_detection
python train_yolo.py
```

## ⚠️ Data Availability Statement

The source code and all extracted data (e.g., CSV and NPY files) used in this study are publicly available in this repository to support reproducibility. Due to privacy considerations, only a subset of the raw video data (MP4 format) is publicly released. The remaining raw videos are available from the corresponding author upon reasonable request.

## 🔗 Citation

If you find this code or our research helpful, please cite our manuscript:

```bash
@article{zhao2025fusion,
  title={Fusion of Skeletal and Racket Dynamics for Enhanced Table Tennis Action Recognition},
  author={Zhao, Ruiqi and Li, Chenlong and Jia, Lingyu},
  journal={The Visual Computer},
  year={2025},
  note={Under Review}
}
```

## 📧 Contact

For any questions regarding the code or dataset, please contact:

Lingyu Jia (Corresponding Author): <jlingyu@cau.edu.cn>
