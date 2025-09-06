# CIFAR-10 CNN Classifier

A PyTorch implementation of a Convolutional Neural Network for image classification on the CIFAR-10 dataset.  
This project demonstrates a compact CNN architecture trained with modern techniques such as data augmentation, AdamW optimizer, label smoothing, dropout, and cosine learning rate scheduling.

## Features
- 4 convolutional blocks with BatchNorm and ReLU
- Global Average Pooling and a simple classifier head
- Data augmentation: random crop, horizontal flip, color jitter
- Regularization: dropout and label smoothing
- Optimizer: AdamW with weight decay
- Cosine Annealing learning rate scheduler
- Training loop with per-epoch train/test accuracy logging
