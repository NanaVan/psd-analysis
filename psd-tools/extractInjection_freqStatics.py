#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import sqlite3, os, glob
from tqdm import tqdm

# --- 配置 (请务必检查路径是否存在) ---
SOURCE_FOLDER = './data_folder'        
BASELINE_FOLDER = './baseline_folder'  
STATS_CSV = 'channel_statistics.csv' 
DB_FILE = 'channel_analysis.db'

# 如果数据库已存在则删除，重新开始
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

# --- 1. 初始化数据库结构 ---
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# 提高写入速度的配置
cursor.execute("PRAGMA synchronous = OFF")
cursor.execute("PRAGMA journal_mode = MEMORY")

# 创建表
cursor.execute("""
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    file_name TEXT
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS frequencies (
    id INTEGER PRIMARY KEY,
    frequency REAL
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS outliers (
    freq_id INTEGER,
    file_id INTEGER,
    level INTEGER, -- 0: over_mean, 1: over_1std, 2: over_3std
    FOREIGN KEY(freq_id) REFERENCES frequencies(id),
    FOREIGN KEY(file_id) REFERENCES files(id)
)""")
conn.commit()

# --- 2. 预填充频率表并载入阈值 ---
print("正在初始化频率与阈值...")
stats_df = pd.read_csv(STATS_CSV)
freq_values = stats_df['frequency'].values
threshold_mean = stats_df['mean'].values
threshold_std1 = threshold_mean + stats_df['std'].values
threshold_std3 = threshold_mean + 3 * stats_df['std'].values

# 批量插入频率索引
cursor.executemany("INSERT INTO frequencies (id, frequency) VALUES (?, ?)", 
                   enumerate(freq_values))
conn.commit()

# --- 3. 逐个文件处理并实时写入 ---
file_paths = sorted(glob.glob(os.path.join(SOURCE_FOLDER, "*.npz")))

print(f"开始处理 {len(file_paths)} 个文件并写入数据库...")
for file_idx, file_path in enumerate(tqdm(file_paths)):
    file_name = os.path.basename(file_path)
    
    # 记录文件名到数据库
    cursor.execute("INSERT INTO files (id, file_name) VALUES (?, ?)", (file_idx, file_name))
    
    baseline_path = os.path.join(BASELINE_FOLDER, f"baseline_{file_name}")
    
    try:
        with np.load(file_path) as data:
            spectrum = data['psd_arrays'].mean(axis=0)
            
        if os.path.exists(baseline_path):
            with np.load(baseline_path) as b_data:
                baseline = b_data['psd_arrays'].mean(axis=0)
                spectrum = spectrum - baseline
        
        # 向量化比对获取索引
        # 逻辑：level 2(3std) 必定属于 level 1(1std)，level 1 必定属于 level 0(mean)
        idx_std3 = np.where(spectrum > threshold_std3)[0]
        idx_std1 = np.where(spectrum > threshold_std1)[0]
        idx_mean = np.where(spectrum > threshold_mean)[0]

        # 准备批量插入数据 (freq_id, file_id, level)
        insert_data = []
        # 注意：为了减小数据库体积，你可以选择只记录最高级别，
        # 或者按照你的需求全部记录。这里按你要求的三个列表逻辑全部记录：
        for f_idx in idx_mean: insert_data.append((int(f_idx), file_idx, 0))
        for f_idx in idx_std1: insert_data.append((int(f_idx), file_idx, 1))
        for f_idx in idx_std3: insert_data.append((int(f_idx), file_idx, 2))
        
        # 每处理一个文件写入一次，防止内存堆积
        cursor.executemany("INSERT INTO outliers VALUES (?, ?, ?)", insert_data)
        
        # 每 100 个文件执行一次 commit，平衡速度与安全
        if file_idx % 100 == 0:
            conn.commit()

    except Exception as e:
        print(f"\n处理 {file_name} 出错: {e}")

# --- 4. 收尾工作 ---
print("\n正在创建索引以优化后续查询...")
cursor.execute("CREATE INDEX idx_freq ON outliers(freq_id)")
cursor.execute("CREATE INDEX idx_level ON outliers(level)")
conn.commit()
conn.close()

print(f"任务完成！数据库文件已生成: {DB_FILE}")

# --- 5. 查询方式 ---
# 你想查频率点 308.3 对应的所有 over_3std 文件
#conn = sqlite3.connect(DB_FILE)
#cursor = conn.cursor()
#cursor.execute("SELECT file_name FROM files JOIN outliers ON files.id = outliers.file_id WHERE freq_id = (SELECT id FROM frequencies WHERE frequency = 308.3) AND level = 2;").fetchall()
