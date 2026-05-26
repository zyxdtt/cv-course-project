# Steel Surface Defect Segmentation with ResNet34‑UNet

This repository contains the complete training and inference code for a deep learning system that detects and segments four types of surface defects on steel sheets. The model uses a **U‑Net architecture with a ResNet34 encoder** pretrained on ImageNet. It is trained on the Severstal Steel Defect Detection dataset.

> 📌 **Model weights are hosted separately on Hugging Face Hub:**  
> [https://huggingface.co/zyxdtt/ResNet34-UNet/tree/main](https://huggingface.co/zyxdtt/ResNet34-UNet/tree/main)

---

## 📝 Problem Statement

Steel manufacturing requires high‑quality surface inspection. Defects such as scratches, inclusions, and rolling marks can affect product strength and appearance. Manual inspection is time‑consuming and inconsistent. This project automates defect detection using a convolutional neural network that outputs pixel‑wise segmentation masks for four defect classes.

---

## 🧠 Model Architecture

The model follows the **U‑Net design** with a ResNet34 encoder:

| Component | Description |
|-----------|-------------|
| **Encoder** | ResNet34 pretrained on ImageNet. Five stages produce feature maps with channels [64, 64, 128, 256, 512] and spatial downsampling factors [2, 4, 8, 16, 32]. |
| **Bottleneck** | Double convolution block (512 → 512). |
| **Decoder** | Four up‑convolution blocks (transposed convolution + double convolution). Each block halves the number of channels, doubles resolution, and concatenates with the corresponding encoder feature map via skip connections. |
| **Output layer** | 1×1 convolution with sigmoid activation producing four probability maps (one per defect class). |

**Skip connections** help retain fine‑grained localization information.

---

## 📊 Training Details (from Designment.pdf)

| Parameter | Value |
|-----------|-------|
| Input image size | 256 × 1600 pixels |
| Training set | 6,666 images (80% train, 20% validation) |
| Loss function | `L = L_BCE + (1 - Dice)` |
| Optimizer | AdamW |
| Initial learning rate | 1×10⁻⁴ |
| Scheduler | ReduceLROnPlateau (factor 0.1, patience 2) |
| Batch size (train) | 4 |
| Batch size (validation) | 8 |
| Epochs | 10 (with early stopping based on validation loss) |
| Data augmentation (training only) | Random horizontal flip (p=0.5) |

> Additional augmentations (rotation, brightness/contrast) can be added to improve performance.

---

## 📈 Evaluation Results (from Test Report.pdf)

The model was evaluated on the validation set (1,333 images) using the **Dice coefficient**:

\[
\mathrm{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}
\]

### Best overall performance

- **Overall Dice score:** 0.6296  
- **Optimal probability threshold:** 0.45 (tuned from 0.3 to 0.7)

### Per‑class Dice scores (threshold = 0.45)

| Defect Class | Dice Score |
|--------------|------------|
| Class 1 | 0.651 |
| Class 2 | 0.624 |
| Class 3 | 0.637 |
| Class 4 | 0.606 |

### Loss curves

- Best validation loss: **0.4358** at epoch 10  
- Training loss decreased from 0.97 → 0.75  
- Validation loss decreased from 0.59 → 0.44  
- Minor instability at epochs 7‑8 due to learning rate scheduler response.

### Threshold sensitivity

The Dice score remains stable (0.6293–0.6296) for thresholds between 0.4 and 0.5, indicating that the model produces highly confident predictions (close to 0 or 1).

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/zyxdtt/cv-course-project.git
cd cv-course-project/Semantic%20Segmentation
2. Install dependencies
bash
pip install -r requirements.txt
3. Download model weights
Download best.pth from the Hugging Face Hub:
https://huggingface.co/zyxdtt/ResNet34-UNet/tree/main
Place it in a weights/ folder inside the repository.

4. Run inference
python
python predict.py --image path/to/steel_sheet.png --threshold 0.45
Example output will be a set of binary masks (one per defect class) overlaid on the input image.

📁 Repository Contents
File	Description
train.py	Training loop with AdamW, learning rate scheduler, and model checkpointing
model.py	U‑Net with ResNet34 encoder definition
dataset.py	Data loader with RLE decoding, resizing, and augmentation
utils.py	Dice coefficient, RLE encoding/decoding, threshold tuning
predict.py	Inference script
requirements.txt	Python dependencies (PyTorch, torchvision, PIL, numpy, etc.)
Designment.pdf	Full system design document
Test Report.pdf	Full experimental results and analysis
📄 License
MIT

🙏 Acknowledgements
Severstal Steel Defect Detection dataset

U‑Net paper: Ronneberger et al., 2015

ResNet: He et al., 2016
