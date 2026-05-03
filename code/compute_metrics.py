# compute_metrics.py
import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# ========== 已适配 GitHub 仓库结构 ==========
pred_dir = "../data/predictions"
gt_dir   = "../data/ground_truth"
output_csv = "../results/baseline_metrics.csv"
# ===========================================

# 确保结果文件夹存在
os.makedirs("../results", exist_ok=True)

# 获取所有预测文件
pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")]

# 初始化指标计算器
dice_metric = DiceMetric(include_background=False, reduction="mean")
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

results = []

for pf in pred_files:
    pred_path = os.path.join(pred_dir, pf)
    base_name = pf.replace(".nii.gz", "")
    gt_path = os.path.join(gt_dir, pf)

    if not os.path.exists(gt_path):
        print(f"警告：找不到 {gt_path}，跳过 {pf}")
        continue

    # 加载数据
    pred_nib = nib.load(pred_path)
    gt_nib   = nib.load(gt_path)
    pred = pred_nib.get_fdata().astype(np.int16)
    gt   = gt_nib.get_fdata().astype(np.int16)

    # 转为 PyTorch tensor
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

print("\n==================================")
print(f"✅ 指标已保存到 {output_csv}")
print(f"📊 平均 Dice = {df['Dice'].mean():.4f}")
print(f"📏 平均 HD95 = {df['HD95_mm'].mean():.2f} mm")
print("==================================")
