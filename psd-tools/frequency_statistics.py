import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置路径
SOURCE_FOLDER = './your_npz_folder/' 
BASELINE_FOLDER = './your_baseline_folder/'
OUTPUT_FILE = 'channel_statistics.csv'

def process_spectrums(folder_path, output_csv, baseline_path=''):
    # 1. 获取所有 .npz 文件路径并排序
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
    files.sort()
    
    num_files = len(files)
    num_channels = len(np.load(files[0])['frequencies'])-1
    mmap_file = 'buffer_matrix.dat' # 临时磁盘映射文件，约 34GB
    
    frequencies = None
    
    # 2. 创建磁盘映射矩阵 (float64)
    # mode='w+' 表示如果不存在则创建，存在则覆盖
    matrix = np.memmap(mmap_file, dtype='float64', mode='w+', shape=(num_channels, num_files))

    print(f"正在读取 {num_files} 个文件...")
    
    for i, file_path in enumerate(tqdm(files)):
        file_name = os.path.basename(file_path)
        baselinefile_path = os.path.join(baseline_path, "baseline_{:}.npy".format(file_name[:-4]))
        try:
            with np.load(file_path) as data:
                # 只在读取第一个文件时提取频率信息
                if frequencies is None:
                    frequencies = data['frequencies'][:-1].astype('float64')
                
                # 对当前文件的 psd_arrays 取均值，得到 (262144,) 的数组
                # 确保使用 float64 以维持量级差异大的数据的精度
                spectrum = np.mean(data['psd_arrays'], axis=0).astype('float64')
                if os.path.exists(baselinefile_path):
                    spectrum = spectrum / np.load(baselinefile_path)
                else:
                    print(f"警告：未找到对应的本底文件 {baselinefile_path}，将跳过本底扣除。")
                
                # 将结果存入磁盘映射矩阵的第 i 列
                matrix[:, i] = spectrum
                
                # 每 100 个文件刷新一次磁盘缓存，防止内存堆积
                if i % 100 == 0:
                    matrix.flush()
        except Exception as e:
            print(f"\n[错误] 处理文件 {os.path.basename(file_path)} 时出错: {e}")

    print("\n所有数据已缓存至磁盘。正在分块计算统计指标...")

    # 3. 分块计算统计量 (避免计算时一次性加载 34GB 到内存)
    # 每次处理 8192 个通道，内存占用极低
    chunk_size = 8192 
    results = []

    for start in tqdm(range(0, num_channels, chunk_size)):
        end = min(start + chunk_size, num_channels)
        
        # 仅将当前 chunk 的数据载入内存
        chunk_data = matrix[start:end, :]
        
        # 计算统计值 (axis=1 表示横向跨文件计算)
        chunk_mean = np.mean(chunk_data, axis=1)
        chunk_std = np.std(chunk_data, axis=1)
        chunk_max = np.max(chunk_data, axis=1)
        chunk_min = np.min(chunk_data, axis=1)
        chunk_median = np.median(chunk_data, axis=1)
        
        # 整理该 block 的结果
        for j in range(len(chunk_mean)):
            global_idx = start + j
            results.append({
                'frequency': frequencies[global_idx],
                'mean': chunk_mean[j],
                'std': chunk_std[j],
                'max': chunk_max[j],
                'min': chunk_min[j],
                'median': chunk_median[j]
            })

    # 4. 导出为 CSV
    print("正在生成 CSV 文件...")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # 5. 清理：必须先删除变量引用，才能安全删除磁盘上的临时文件
    del matrix
    if os.path.exists(mmap_file):
        os.remove(mmap_file)
        print(f"临时映射文件已清理。任务完成！结果保存在: {output_csv}")

process_spectrums(SOURCE_FOLDER, OUTPUT_FILE, BASELINE_FOLDER)
