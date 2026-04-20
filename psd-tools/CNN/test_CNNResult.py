#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import os

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
            nn.AdaptiveMaxPool1d(256)) # -> 2048
            
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
    TARGET_LENGTH = 209715  # 与你训练时的序列点数保持一致
    CLASS_NAMES = {0: "纯本底 (Background)", 1: "含有信号 (Signal)"}

    # 2. 加载模型
    device = torch.device("cpu")
    model = SpectrumCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    if not os.path.exists(TEST_DIR):
        print(f"错误：未找到测试目录 {TEST_DIR}")
        return

    # 3. 遍历并预测
    print(f"{'文件名':<30} | {'判定类别':<20} | {'置信度'}")
    print("-" * 70)

    files = [f for f in os.listdir(TEST_DIR) if f.endswith('.npz')]
    
    if not files:
        print("测试目录下没有找到 .npz 文件。")
        return

    with torch.no_grad():
        for f in files:
            path = os.path.join(TEST_DIR, f)
            # 加载并处理数据
            try:
                with np.load(path) as loader:
                    avg_psd = np.mean(loader['psd_arrays'], axis=0).astype(np.float32)
                    if len(avg_psd) > TARGET_LENGTH:
                        avg_psd = avg_psd[:TARGET_LENGTH]
                    elif len(avg_psd) < TARGET_LENGTH:
                        avg_psd = np.pad(avg_psd, (0, TARGET_LENGTH - len(avg_psd)))

                    avg_psd = 10 * np.log10(avg_psd)
                    mean_val = np.mean(avg_psd)
                    std_val = np.std(avg_psd)
                    ch1 = torch.from_numpy((avg_psd - mean_val) / std_val).float()

                    x_tensor = torch.from_numpy(avg_psd).float().unsqueeze(0).unsqueeze(0)
                    padding = 1001//2
                    bg_trend = nn.functional.avg_pool1d(x_tensor, kernel_size=1001, stride=1, padding=padding)
                    ch2 = x_tensor - bg_trend[..., :TARGET_LENGTH]
                    ch2 = ch2.squeeze()
                    ch2 = ch2 / ch2.std()
                                
                # 转换维度 (Batch=1, Channel=1, Length)
                input_tensor = torch.stack([ch1, ch2], dim=0).unsqueeze(0).to(device)
                
                # 推理
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                class_id = predicted.item()
                conf_val = confidence.item()
                
                print(f"{f:<30} | {CLASS_NAMES[class_id]:<20} | {conf_val:.2%}")
            
            except Exception as e:
                print(f"处理文件 {f} 时出错: {e}")

if __name__ == "__main__":
    run_inference()
