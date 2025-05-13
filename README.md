# Parameter Efficient Supervised Fine-Tuning

## Introduction

Development status: ongoing

Provide an open-source library which implements multiple PEFT (parameter efficient fine tuning) methods, including [gradient based SMT](https://openreview.net/pdf?id=GbgCRJedQ7), [saliency map](https://arxiv.org/pdf/1312.6034), and [LORA](https://arxiv.org/pdf/2106.09685), with HuggingFace’ TRL library.

*  First to propose a saliency map–based PEFT approach, enabling interpretable parameter-efficient tuning.
*  Ported the [SMT (sparse matrix matrices) implementation](https://github.com/HectorHHZ/Sparse_Matrix_Tuning) from DeepSpeed to Hugging Face’s TRL library for improved integration and maint

## Related Paper

*  He, Haoze, Li, Juncheng Billy, Jiang, Xuan, and Miller, Heather. SMT: Fine-Tuning Large Language Models with Sparse Matrices. ICLR 2025.
*  Hu, Edward J., et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022
*  Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. arXiv preprint arXiv:1312.6034 (2013).

