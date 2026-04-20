#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import os
import csv  # 新增：用于导出CSV

# 必须保证模型定义与训练时完全一致
class SpectrumCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SpectrumCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1024))

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(256))
            
        self.fc = nn.Sequential(
            nn.Linear(64 * 256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def run_inference():
    # 1. 配置路径与参数
    MODEL_PATH = "model_cpu_v1.pth"
    TEST_DIR = "C:/Users/van4w/Desktop/127La/test_CNN/test"
    OUTPUT_CSV = "inference_results.csv"  # 新增：输出文件名
    TARGET_LENGTH = 209715
    
    # 2. 加载模型
    device = torch.device("cpu")
    model = SpectrumCNN()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print(f"错误：未找到模型权重文件 {MODEL_PATH}")
        return

    if not os.path.exists(TEST_DIR):
        print(f"错误：未找到测试目录 {TEST_DIR}")
        return

    files = [f for f in os.listdir(TEST_DIR) if f.endswith('.npz')]
    if not files:
        print("测试目录下没有找到 .npz 文件。")
        return

    # 用于存储所有结果的列表
    results = []

    # 3. 遍历并预测
    print(f"{'文件名':<30} | {'判定结果':<10} | {'置信度'}")
    print("-" * 60)

    with torch.no_grad():
        for f in files:
            path = os.path.join(TEST_DIR, f)
            try:
                with np.load(path) as loader:
                    # 获取数据并预处理
                    avg_psd = np.mean(loader['psd_arrays'], axis=0).astype(np.float32)
                    if len(avg_psd) > TARGET_LENGTH:
                        avg_psd = avg_psd[:TARGET_LENGTH]
                    elif len(avg_psd) < TARGET_LENGTH:
                        avg_psd = np.pad(avg_psd, (0, TARGET_LENGTH - len(avg_psd)))

                    # 通道 1: Log10 + Z-Score
                    avg_psd_log = 10 * np.log10(avg_psd + 1e-12) # 防止log(0)
                    mean_val = np.mean(avg_psd_log)
                    std_val = np.std(avg_psd_log)
                    ch1 = torch.from_numpy((avg_psd_log - mean_val) / (std_val + 1e-8)).float()

                    # 通道 2: 去背景增强
                    x_tensor = torch.from_numpy(avg_psd_log).float().unsqueeze(0).unsqueeze(0)
                    padding = 1001 // 2
                    bg_trend = nn.functional.avg_pool1d(x_tensor, kernel_size=1001, stride=1, padding=padding)
                    ch2 = x_tensor - bg_trend[..., :TARGET_LENGTH]
                    ch2 = ch2.squeeze()
                    ch2 = ch2 / (ch2.std() + 1e-8)
                                
                # 组合输入 (Batch=1, Channel=2, Length)
                input_tensor = torch.stack([ch1, ch2], dim=0).unsqueeze(0).to(device)
                
                # 推理
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                class_id = predicted.item()
                conf_val = confidence.item()
                
                # 打印实时进度
                print(f"{f:<30} | {class_id:<10} | {conf_val:.2%}")
                
                # 记录结果 (文件名, 类别ID, 置信度)
                results.append([f, class_id, f"{conf_val:.4f}"])
            
            except Exception as e:
                print(f"处理文件 {f} 时出错: {e}")

    # 4. 保存为 CSV 文件
    try:
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['filename', 'label', 'confidence'])
            # 写入数据
            writer.writerows(results)
        print("-" * 60)
        print(f"推理完成，结果已保存至: {os.path.abspath(OUTPUT_CSV)}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")

if __name__ == "__main__":
    run_inference()
