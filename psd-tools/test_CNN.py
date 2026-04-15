#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

# 1. 定义与之前讨论一致的 1D-CNN 模型架构
class SpectrumCNN(nn.Module):
    def __init__(self):
        super(SpectrumCNN, self).__init__()
        # 使用较大的卷积核 (k=15) 来自动学习如何滤除平缓的本底
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # 全局最大池化：捕捉整段频谱中最突出的尖峰特征
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出 0~1 之间的概率得分
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. 数据预处理函数
def preprocess_spectrum(data_array):
    """
    对原始频谱进行标准化，增强模型对不同量级信号的鲁棒性
    """
    # 确保是 float32 类型
    data = data_array.astype(np.float32)
    # 简单的 Z-score 标准化：减去均值，除以标准差
    # 这有助于 CNN 关注局部“突变”而非绝对数值大小
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        data = (data - mean) / std
    
    # 转换为 PyTorch Tensor 并增加通道维 (Batch, Channel, Length)
    # 1D-CNN 需要输入维度为 (N, 1, L)
    tensor_data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor_data

# 3. 推理与得分函数
def get_classification_scores(data_0, data_1, model_path=None):
    model = SpectrumCNN()
    model.eval()
    
    # 如果你有训练好的权重，可以取消下面这一行的注释
    # model.load_state_dict(torch.load(model_path))
    
    with torch.no_grad():
        # 处理纯本底数据
        t_0 = preprocess_spectrum(data_0)
        score_0 = model(t_0).item()
        
        # 处理有峰信号数据
        t_1 = preprocess_spectrum(data_1)
        score_1 = model(t_1).item()
        
    print("-" * 30)
    print(f"纯本底数据 (data_0) 的分类得分: {score_0:.4f}")
    print(f"有峰信号数据 (data_1) 的分类得分: {score_1:.4f}")
    print("-" * 30)
    
    if score_1 > score_0:
        print(f"建议分类阈值设定在: {(score_0 + score_1) / 2:.4f}")
    else:
        print("警告：当前模型权重可能尚未针对你的数据进行训练。")
    
    return score_0, score_1

# 使用示例 (假设你已经有了 data_0 和 data_1 数组)
output_folder_0 = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/'
file_strs_0 = ['8243_PY82ch1_0000_trigger_1_2026-04-07T22-04-29.npz', '8243_PY82ch1_0000_trigger_10_2026-04-07T22-04-56.npz', '8243_PY82ch1_0000-0001_2026-04-07T22-05-20.npz']
output_folder_1 = 'C:/Users/van4w/Desktop/127La/8241_result/'
file_strs_1 = ['8241_PY82ch1_0011_trigger_1_2026-04-06T04-42-00.npz', '8241_PY82ch1_0012_trigger_17_2026-04-06T04-43-42.npz', '8241_PY82ch1_0011-0012_2026-04-06T04-42-51.npz']

data_0 = np.mean(np.load(output_folder_0+file_strs_0[0])['psd_arrays'], axis=0)
data_1 = np.mean(np.load(output_folder_1+file_strs_1[0])['psd_arrays'], axis=0)
get_classification_scores(data_0, data_1)

