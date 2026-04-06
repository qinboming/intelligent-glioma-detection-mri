# intelligent-glioma-detection-mri
Intelligent detection of glioma based on MRI using deep learning methods

## 📌 项目简介
本项目基于多模态MRI影像，采用深度学习技术实现脑胶质瘤的智能检测、分割与分级，为临床辅助诊断提供技术支持。

## 📁 仓库结构

## 📚 核心文献（5篇，均含开源代码）
1. **3D U-Net** (MICCAI 2016) - 3D分割基线
2. **nnU-Net** (Nature Methods 2021) - SOTA分割框架
3. **TransBTS** (MICCAI 2021) - Transformer分割模型
4. **BraTS 2020 Challenge** (MICCAI 2020) - 基准数据集
5. **DeepMedic** (IEEE TMI 2016) - 轻量级检测模型

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
