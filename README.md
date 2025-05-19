# Code for ListenNet
[PyTorch](https://pytorch.org/) implementation on: A Lightweight Spatio-Temporal Enhancement Nested Network for Auditory Attention Detection.

## Introduction
We propose a **Li**ghtweight **S**patio-**T**emporal  **E**nhancement **N**ested **Net**work (ListenNet) with low parameters and complexity. It captures multi-channel dependencies, multi-scale dynamic temporal patterns, and multi-step spatio-temporal dependencies, ensuring high accuracy and strong generalization. Experimental results on three public datasets demonstrate the superiority of ListenNet over state-of-the-art methods in both subject-dependent and challenging subject-independent settings, while reducing the trainable parameter count by approximately 7 times.

Cunhang Fan, Xiaoke Yang, Hongyu Zhang, Ying Chen, Lu Li, Jian Zhou, Zhao Lv.ListenNet: A Lightweight Spatio-Temporal Enhancement Nested Network for Auditory Attention Detection. In IJCAI 2025.

<p align="center">
<img src="https://github.com/fchest/ListenNet/blob/main/OVERVIEW.png">
</p>

# Preprocess
* Please download the AAD dataset for training.
* The public [KUL dataset](https://zenodo.org/records/4004271), [DTU dataset](https://zenodo.org/record/1199011#.Yx6eHKRBxPa) and [AVED dataset](https://iiphci.ahu.edu.cn/toAuditoryAttention) are used in this paper.

# Requirements
+ Python 3.10 \
`pip install -r requirements.txt`

# Run
* Using dep/main.py to train and test the model in sub_dependent(within-trial 8:1:1) setting
* Using indep/main.py to train and test the model in sub_independent(Leave-one-subject-out) setting

If you find the paper or this repo useful, please cite
```
@article{fan2025listennet,
  title={ListenNet: A Lightweight Spatio-Temporal Enhancement Nested Network for Auditory Attention Detection},
  author={Fan, Cunhang and Yang, Xiaoke and Zhang, Hongyu and Chen, Ying and Li, Lu and Zhou, Jian and Lv, Zhao},
  journal={arXiv preprint arXiv:2505.10348},
  year={2025}
}
```
