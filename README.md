# intelligent-glioma-detection-mri
Intelligent detection of glioma based on MRI using deep learning methods

## 📌 项目简介
本项目基于多模态MRI影像，采用深度学习技术实现脑胶质瘤的智能检测、分割与分级，为临床辅助诊断提供技术支持。

## 📁 仓库结构


## 📚 核心文献（5篇，均含开源代码）
1. 3D U-Net (MICCAI 2016) - 3D分割基线，[代码仓库](sslocal://flow/file_open?url=https%3A%2F%2Fgithub.com%2Fwolny%2Fpytorch-3dunet&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
2. nnU-Net (Nature Methods 2021) - SOTA分割框架，[代码仓库](sslocal://flow/file_open?url=https%3A%2F%2Fgithub.com%2FMIC-DKFZ%2FnnUNet&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
3. TransBTS (MICCAI 2021) - Transformer分割模型，[代码仓库](sslocal://flow/file_open?url=https%3A%2F%2Fgithub.com%2FWenxuan-1997%2FTransBTS&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
4. BraTS 2020 Challenge (MICCAI 2020) - 基准数据集，[代码仓库](sslocal://flow/file_open?url=https%3A%2F%2Fgithub.com%2FNeuroAI-lab%2FBraTS2020&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
5. DeepMedic (IEEE TMI 2016) - 轻量级检测模型，[代码仓库](sslocal://flow/file_open?url=https%3A%2F%2Fgithub.com%2FKamnitsask%2Fdeepmedic&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
## 🧪 实验规范
- 实验记录统一存放于 `docs/experiment/`
- 文献笔记存放于 `docs/papers/`
- 模型代码存放于 `code/`
- 数据集不提交至GitHub，仅说明来源

## 🛠️ 环境依赖
- Python 3.8+
- PyTorch 1.10+
- MONAI 1.0+
- SimpleITK
- NumPy, Pandas, Matplotlib

## 📊 数据集
- 公开数据集：BraTS 2020/2021（https://www.med.upenn.edu/cbica/brats2020/）
- 模态：T1、T1ce、T2、FLAIR 四种MRI序列

## 🤝 维护者
- 维护者：qinboming
- 更新时间：2026-04-07
