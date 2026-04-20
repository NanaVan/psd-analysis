#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class SpectrumDataset(Dataset):
    def __init__(self, data_list, labels, target_length=209715, is_train=True):
        self.data_list = data_list
        self.labels = labels
        self.target_length = target_length
        self.is_train = is_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data_list[idx].copy().astype(np.float32)
        
        # 1. 转换为对数域（dB）
        data = 10 * np.log10(data)
        
        # 2. 通道1： Z-Score 归一化（局部归一化）
        mean_val = np.mean(data)
        std_val = np.std(data)
        ch1 = torch.from_numpy((data - mean_val) / std_val).float()

        # 3. 通道2：局部滑动平均去本底
        padding = 1001 // 2
        x_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0) # [1,1,L]
        bg_trend = nn.functional.avg_pool1d(x_tensor, kernel_size=1001, stride=1, padding=padding)
        ch2 = x_tensor - bg_trend[...,:x_tensor.shape[-1]]
        ch2 = ch2.squeeze()
        ch2 = ch2 / ch2.std()

        x_dual = torch.stack([ch1, ch2], dim=0)
        return x_dual, self.labels[idx]
        

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

def load_npz_data(directory):
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []
    files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    data_list = []
    for f in files:
        path = os.path.join(directory, f)
        # 你的逻辑：读取 psd_arrays 并取均值
        with np.load(path) as loader:
            avg_psd = np.mean(loader['psd_arrays'], axis=0)
            data_list.append(avg_psd)
    return data_list

def main():
    # 1. 设置路径
    class0_dir = "C:/Users/van4w/Desktop/127La/test_CNN/0"
    class1_dir = "C:/Users/van4w/Desktop/127La/test_CNN/1"
    
    # 2. 加载
    print("正在加载 .npz 数据...")
    data0 = load_npz_data(class0_dir)
    data1 = load_npz_data(class1_dir)
    
    X = data0 + data1
    y = [0]*len(data0) + [1]*len(data1)
    
    if len(X) == 0:
        print("未加载到数据，请检查路径。")
        return

    # 3. 划分与 Dataset
    # 自动获取第一个文件的长度作为 target_length
    data_len = X[0].shape[-1] 
    print(f"数据总数: {len(X)}, 序列点数: {data_len}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(SpectrumDataset(X_train, y_train, data_len), batch_size=8, shuffle=True)
    val_loader = DataLoader(SpectrumDataset(X_val, y_val, data_len), batch_size=8)

    # 4. 训练 (强制 CPU 模式)
    device = torch.device("cpu")
    model = SpectrumCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("开始 CPU 训练...")
    for epoch in range(40):
        model.train()
        for i, (imgs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} 完成")

    torch.save(model.state_dict(), "model_cpu_v1.pth")
    print("模型已保存。")

if __name__ == "__main__":
    main()
