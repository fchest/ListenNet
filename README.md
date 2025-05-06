# Code for ListenNet
[PyTorch](https://pytorch.org/) implementation on: A Lightweight Spatio-Temporal Enhancement Nested Network for Auditory Attention Detection.
By Cunhang Fan, Xiaoke Yang, Hongyu Zhang, Ying Chen, Lu Li, Jian Zhou and Zhao Lv*
## Introduction
We propose a **Li**ghtweight **S**patio-**T**emporal  **E**nhancement **N**ested **Net**work (ListenNet) with low parameters and complexity. It captures multi-channel dependencies, multi-scale dynamic temporal patterns, and multi-step spatio-temporal dependencies, ensuring high accuracy and strong generalization. Experimental results on three public datasets demonstrate the superiority of ListenNet over state-of-the-art methods in both subject-dependent and challenging subject-independent settings, while reducing the trainable parameter count by approximately 8 times.


<p align="center">
<img src="https://github.com/fchest/ListenNet/blob/main/OVERVIEW.png">
</p>

# Preprocess
* Please download the AAD dataset for training.
* The public [KUL dataset](https://zenodo.org/records/4004271), [DTU dataset](https://zenodo.org/record/1199011#.Yx6eHKRBxPa) and [AVED dataset](https://iiphci.ahu.edu.cn/toAuditoryAttention) are used in this paper.

# Requirements
+ Python3.9 \
`pip install -r requirements.txt`

# Run
* Using main.py to train and test the model
