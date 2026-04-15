#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 配置参数
DATA_DIR = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/'  # npz文件存放路径
OUTPUT_CSV = 'labels.csv'    # 结果保存路径

class DataLabeler:
    def __init__(self, data_dir, output_csv):
        self.data_dir = data_dir
        self.output_csv = output_csv
        
        # 检查目录是否存在
        if not os.path.exists(data_dir):
            print(f"错误: 找不到文件夹 {data_dir}")
            return

        # 获取所有npz文件并排序
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        
        # 加载进度
        self.results = self._load_progress()
        self.remaining_files = [f for f in all_files if f not in self.results]
        self.current_idx = 0
        
        if not self.remaining_files:
            print("所有文件已标注完成或文件夹为空！")
            return

        print(f"待处理文件数: {len(self.remaining_files)}")

        # 初始化绘图窗口
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title('数据标注工具 - 按 0/1 标注, Q 退出')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.show_next()
        plt.show()

    def _load_progress(self):
        """读取CSV，获取已标注的文件名"""
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv)
                return dict(zip(df['filename'], df['label']))
            except Exception:
                return {}
        return {}

    def save_result(self, filename, label):
        """保存标注结果"""
        file_exists = os.path.exists(self.output_csv)
        df = pd.DataFrame([[filename, label]], columns=['filename', 'label'])
        # 使用 append 模式实时写入，防止崩溃丢失数据
        df.to_csv(self.output_csv, mode='a', index=False, header=not file_exists)
        self.results[filename] = label

    def show_next(self):
        """显示下一个待处理的文件"""
        if self.current_idx >= len(self.remaining_files):
            print("\n恭喜！所有文件处理完毕。")
            plt.close()
            return

        filename = self.remaining_files[self.current_idx]
        path = os.path.join(self.data_dir, filename)
        
        try:
            # 加载并计算均值
            with np.load(path) as data:
                # 针对 shape=(255, 262143) 取均值
                freq = data['frequencies'][:-1]
                mean_psd = np.mean(data['psd_arrays'], axis=0)
            
            self.ax.clear()
            self.ax.plot(freq, mean_psd, linewidth=0.8, color='steelblue')
            self.ax.set_xlabel('Frequency [Hz]')
            self.ax.set_xlim((freq[0], freq[-1]))
            self.ax.set_yscale('log')  # 对数坐标
            
            # 修正了之前报错的地方：添加了 self.
            title = (f"[{self.current_idx + 1}/{len(self.remaining_files)}] {filename}\n"
                     f"Keys: [0] Class 0 for only background | [1] Class 1 for signals with background | [Q] Quit")
            self.ax.set_title(title)
            self.ax.set_ylabel('Mean PSD')
            self.ax.set_xlabel('Frequencies Index')
            self.ax.grid(True, which='both', alpha=0.3)
            
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")
            self.current_idx += 1
            self.show_next()

    def on_key(self, event):
        """处理键盘输入"""
        if event.key is None:
            return
            
        key = event.key.lower()
        
        if key in ['0', '1']:
            filename = self.remaining_files[self.current_idx]
            label = int(key)
            self.save_result(filename, label)
            print(f"进度: {self.current_idx + 1}/{len(self.remaining_files)} | {filename} -> {label}")
            
            self.current_idx += 1
            self.show_next()
            
        elif key == 'q':
            print("正在退出...")
            plt.close()

if __name__ == "__main__":
    # 执行
    labeler = DataLabeler(DATA_DIR, OUTPUT_CSV)
