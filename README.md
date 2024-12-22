# Mammals Image Classification

## Overview
This project involves building a Convolutional Neural Network (CNN) to classify images of mammals into two categories: **Arctic Fox** and **Elephant**. The model is trained, validated, and tested using a dataset curated from Kaggle. The primary goal is to demonstrate effective image classification using deep learning techniques, hyperparameter tuning, and transfer learning.

---

## Dataset

### Description
- The dataset contains images of various mammals, with a subset focusing on **Arctic Fox** and **Elephant**.
- Images were created using Stable Diffusion
- Images are labeled and organized for supervised learning tasks.
- The dataset includes samples with diverse environments and poses, adding complexity and richness to the training process.

### Dataset Statistics
- **Training Set**: 640 images
- **Validation Set**: 161 images
- **Testing Set**: 200 images
- **Image Size**: Resized to 224x224 pixels

### Data Normalization
- **Mean**: [0.5059, 0.5021, 0.5020]
- **Standard Deviation**: [0.2108, 0.1982, 0.1949]

### Data Augmentation Techniques
- **Resize**: 224x224
- **Random Horizontal Flip**
- **Random Rotation**: ±10°

### Dataset Link
The dataset can be accessed [here](https://drive.google.com/drive/folders/1MQ3615sT-IVFSDjIbZTb2MH1O3kKJDzi?usp=sharing).

---

## Neural Network Architecture

### Simple CNN
1. **Input Layer**:
   - Accepts RGB images resized to 224x224.
   - 3 input channels.
2. **Convolutional Layers**:
   - 4 convolutional blocks.
   - Each block includes:
     - **Convolutional Layer**: 32 filters of size 3x3 with padding.
     - **ReLU Activation**: Introduces non-linearity.
     - **MaxPooling**: Down-samples spatial dimensions by 2.
3. **Fully Connected Layer**:
   - Outputs probabilities for 2 classes.

### Transfer Learning with ResNet-18
- **Pretrained Weights**: ImageNet (IMAGENET1K_V1)
- **Fine-Tuned Classes**: Arctic Fox and Elephant
- Achieved **100% Test Accuracy**

---

## Training and Optimization

### Hyperparameters
- **Mini-batch Size**: 64
- **Optimization Algorithm**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 1e-3
- **Momentum**: 0.9
- **Weight Decay**: 1e-4
- **Learning Rate Decay**: 0.99 (Exponential Decay with `torch.optim.lr_scheduler.ExponentialLR`)

### Results (Simple CNN)
- **Training Accuracy**: 84.52%
- **Training Loss**: 0.57018
- **Testing Accuracy**: 70%
- **Precision**: 81%
- **Recall**: 70%
- **F1 Score**: 67%

### Hyperparameter Tuning
1. **Tool Used**: ClearML for automated tracking and tuning.
2. **Strategy**: Random Search
3. **Optimal Values**:
   - Number of Layers: 4
   - Number of Filters: 32
   - Kernel Size: 3
   - Learning Rate: 0.001
   - Learning Rate Decay: 0.99
   - Momentum: 0.9
   - Weight Decay: 1e-4

---

## Transfer Learning

### Advantages
- Superior feature representation.
- Faster convergence.
- Enhanced accuracy.

---

## Results
| Metric                  | Simple CNN | ResNet-18 Transfer Learning |
|-------------------------|------------|-----------------------------|
| Test Accuracy           | 70%        | 100%                       |
| Precision               | 81%        | -                          |
| Recall                  | 70%        | -                          |
| F1 Score                | 67%        | -                          |

---

## Tools and Frameworks
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Visualization**: Matplotlib
- **Automated Tuning**: ClearML

---

## Future Improvements
- Experiment with deeper architectures like ResNet-50.
- Incorporate advanced augmentation techniques to increase model robustness.
- Explore attention mechanisms for better feature extraction.

---

## Authors
- **Shaagun Suresh**
