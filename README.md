# AnytimeCL: Anytime Continual Learning for Open Vocabulary Classification

## ECCV 2024 Oral

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://eccv2024.ecva.net/)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](http://zzhu.vision/anytime_continual_learning/)
[![YouTube Video](https://img.shields.io/badge/YouTube-Video-red)](https://youtu.be/gSOpLxQi8jg)

## Authors
[Zhen Zhu](https://zzhu.vision) · [Yiming Gong](https://github.com/nickgong1) · [Derek Hoiem](http://dhoiem.cs.illinois.edu)

## Overview

We propose an approach for anytime continual learning (AnytimeCL) for open vocabulary image classification. The AnytimeCL problem aims to break away from batch training and rigid models by requiring that a system can predict any set of labels at any time and efficiently update and improve when receiving one or more training samples at any time. Despite the challenging goal, we achieve substantial improvements over recent methods. We propose a dynamic weighting between predictions of a partially fine-tuned model and a fixed open vocabulary model that enables continual improvement when training samples are available for a subset of a task's labels. We also propose an attention-weighted PCA compression of training features that reduces storage and computation with little impact to model accuracy. Our methods are validated with experiments that test flexibility of learning and inference.

## Hardware
We test our code on a single NVIDIA RTX 3090Ti GPU.

## Installation

### Prerequisites
- Anaconda or Miniconda
- Git

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/jessemelpolio/AnytimeCL.git
   cd AnytimeCL
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate AnytimeCL
   ```

3. Clone the DINOv2 repository:
   ```
   git clone https://github.com/facebookresearch/dinov2.git
   ```


## Project Structure
- `data/`: Dataset handling and preprocessing
- `encode_features/`: Scripts for encoding features using CLIP and DINO
- `engines/`: Engine implementations for training and evaluation
- `models/`: Model architectures and components
- `options/`: Command-line argument parsing
- `scripts/`: Utility scripts
- `main.py`: Main entry point for running experiments

## Usage
1. **Prepare datasets:**
   Our project uses various datasets for target tasks and zero-shot tasks.

   <details>
   <summary>Click to expand dataset details</summary>

   **Target Tasks:** CIFAR100, SUN397, EuroSAT, OxfordIIITPet, Flowers102, FGVCAircraft, StanfordCars, Food101

   **Zero-shot Tasks:** ImageNet, UCF101, DTD

   > **Note:** SUN397, EuroSAT, UCF101, and ImageNet require manual downloading from their original sources. Please follow the instructions in [`tutorials/download_data.md`](tutorials/download_data.md) to obtain these datasets. Other datasets can be easily downloaded using the `torchvision.datasets` package. We also provide additional datasets in the `data/` folder for your convenience but be aware that they are not tested rigorously and may not work with the codebase.
   </details>

   To encode the intermediate image representations of these datasets to speed up training, check the script in [`scripts/encode_features.sh`](scripts/encode_features.sh). After setting the correct data root in the script, you can run the script with:
   ```
   bash scripts/encode_features.sh
   ```

2. **Train:**
   Example scripts for task, data, and class-incremental learning:
   <details>
   <summary>Click to expand example scripts</summary>

   ```
   bash scripts/task_incremental.sh
   ```
   ```
   bash scripts/data_incremental.sh
   ```
   ```
   bash scripts/class_incremental.sh
   ```
   </details>

3. **(Optional) Compress:**
   To compress the features, run the script in [`scripts/compress_features.sh`](scripts/compress_features.sh).
   ```
   bash scripts/compress_features.sh
   ```

## Warning
This codebase is only tested under a single GPU. If you want to use multiple GPUs, you need to modify the codebase. 

We'd appreciate it if you could report any issues you encounter.


## Configuration Options

Our approach offers various customization options to create different experimental settings. Refer to [`tutorials/configuration_options.md`](tutorials/configuration_options.md) for more details.


## Bibtex

If you use this code for your research, please consider citing:
```
@inproceedings{zhu2024anytimecl,
  title={Anytime Continual Learning for Open Vocabulary Classification},
  author={Zhu, Zhen and Gong, Yiming and Hoiem, Derek},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Acknowledgements
- This project uses [DINOv2](https://github.com/facebookresearch/dinov2) by Facebook Research.
- The project incorporates [CLIP](https://github.com/openai/CLIP) for vision-language learning.
- The arguments configuration is inspired from [SPADE](https://github.com/NVlabs/SPADE).

