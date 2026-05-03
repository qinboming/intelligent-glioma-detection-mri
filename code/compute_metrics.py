# compute_metrics.py
import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# ========== 请根据你的实际路径修改下面三个变量 ==========
pred_dir = "/home/yourname/intelligent-glioma-detection-mri/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__2d_fold0/validation"
gt_dir   = "/home/yourname/intelligent-glioma-detection-mri/data/BraTS2021/raw_data/labels"
output_csv = "/home/yourname/intelligent-glioma-detection-mri/docs/experiment/baseline_metrics.csv"
# =====================================================

# 获取所有预测文件（注意扩展名可能是 .nii.gz）
pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")]

# 初始化指标计算器（多类平均，不包括背景）
dice_metric = DiceMetric(include_background=False, reduction="mean")
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

results = []

for pf in pred_files:
    # 预测文件完整路径
    pred_path = os.path.join(pred_dir, pf)
    # 根据预测文件名构造对应的真实标签文件名（假设 GT 文件多了 _seg）
    base_name = pf.replace(".nii.gz", "")
    gt_path = os.path.join(gt_dir, f"{base_name}_seg.nii.gz")

    if not os.path.exists(gt_path):
        print(f"警告：找不到 {gt_path}，跳过 {pf}")
        continue

    # 加载数据
    pred_nib = nib.load(pred_path)
    gt_nib   = nib.load(gt_path)
    pred = pred_nib.get_fdata().astype(np.int16)
    gt   = gt_nib.get_fdata().astype(np.int16)

    # 转为 PyTorch tensor (1, 1, H, W, D)
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    gt_tensor   = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

    # 计算 Dice
    dice_metric(y_pred=pred_tensor, y=gt_tensor)
    dice = dice_metric.aggregate().item()
    dice_metric.reset()

    # 计算 HD95
    hd95_metric(y_pred=pred_tensor, y=gt_tensor)
    hd95 = hd95_metric.aggregate().item()
    hd95_metric.reset()

    results.append({
        "case": base_name,
        "Dice": dice,
        "HD95_mm": hd95
    })
    print(f"{base_name}: Dice={dice:.4f}, HD95={hd95:.2f} mm")

# 保存结果到 CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n指标已保存到 {output_csv}")
print(f"平均 Dice = {df['Dice'].mean():.4f} ± {df['Dice'].std():.4f}")
print(f"平均 HD95 = {df['HD95_mm'].mean():.2f} ± {df['HD95_mm'].std():.2f} mm")
