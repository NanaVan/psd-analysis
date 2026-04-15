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
        
        # 1. 检查目录
        if not os.path.exists(data_dir):
            print(f"错误: 找不到文件夹 {data_dir}")
            return

        # 2. 获取所有文件
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        
        # 3. 加载已有进度 (核心断点续传逻辑)
        self.labeled_filenames = self._get_already_labeled()
        
        # 4. 过滤掉已经标注过的文件
        self.remaining_files = [f for f in all_files if f not in self.labeled_filenames]
        
        print("-" * 30)
        print(f"总文件数:   {len(all_files)}")
        print(f"已标注数:   {len(self.labeled_filenames)}")
        print(f"剩余待办:   {len(self.remaining_files)}")
        print("-" * 30)

        if not self.remaining_files:
            print("所有文件已处理完毕！")
            return

        # 5. 初始化绘图窗口
        self.current_idx = 0
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title('数据标注工具 - 断点续传模式')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.show_next()
        plt.show()

    def _get_already_labeled(self):
        """从CSV中读取已经处理过的文件名集合"""
        if os.path.exists(self.output_csv):
            try:
                # 只读取文件名列，提高效率
                df = pd.read_csv(self.output_csv)
                if 'filename' in df.columns:
                    return set(df['filename'].tolist())
            except Exception as e:
                print(f"读取进度文件失败，将从头开始: {e}")
        return set()

    def save_result(self, filename, label):
        """追加保存结果"""
        file_exists = os.path.exists(self.output_csv)
        df = pd.DataFrame([[filename, label]], columns=['filename', 'label'])
        # 采用 append 模式，不写 index，根据文件是否存在决定是否写 header
        df.to_csv(self.output_csv, mode='a', index=False, header=not file_exists)

    def show_next(self):
        """显示剩余队列中的下一个文件"""
        if self.current_idx >= len(self.remaining_files):
            print("\n恭喜！本批次所有文件处理完毕。")
            plt.close()
            return

        filename = self.remaining_files[self.current_idx]
        path = os.path.join(self.data_dir, filename)
        
        try:
            with np.load(path) as data:
                freq = data['frequencies'][:-1]
                mean_psd = np.mean(data['psd_arrays'], axis=0)
            
            self.ax.clear()
            self.ax.plot(freq, mean_psd, linewidth=0.7, color='steelblue')
            self.ax.set_xlabel('Frequency [Hz]')
            self.ax.set_xlim(freq[0], freq[-1])
            self.ax.set_yscale('log')
            
            # 这里的进度显示是相对于“剩余待办”的
            progress_str = f"Remaining process: {self.current_idx + 1}/{len(self.remaining_files)}"
            self.ax.set_title(f"{progress_str}\nFile: {filename}")
            self.ax.set_ylabel('Mean PSD')
            self.ax.grid(True, which='both', alpha=0.3)
            
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"\n跳过损坏文件 {filename}: {e}")
            self.current_idx += 1
            self.show_next()

    def on_key(self, event):
        if event.key is None: return
        key = event.key.lower()
        
        if key in ['0', '1']:
            filename = self.remaining_files[self.current_idx]
            label = int(key)
            
            # 立即保存到磁盘
            self.save_result(filename, label)
            
            # 终端简易反馈
            print(f"[{self.current_idx+1}] {filename} -> {label} (已保存)", end='\r')
            
            self.current_idx += 1
            self.show_next()
            
        elif key == 'q':
            print("\n检测到退出指令，进度已安全保存。")
            plt.close()

if __name__ == "__main__":
    labeler = DataLabeler(DATA_DIR, OUTPUT_CSV)
