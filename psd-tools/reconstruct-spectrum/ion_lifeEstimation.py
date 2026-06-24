#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps

def run_bayesian_estimation():
    # ==================== 1. 输入您的实验数据 ====================
    # 请在此处填入您整理好的 13 个衰变发生时刻（单位：秒）
    # 对于 A 和 B 测得相差 50ms 的那次事件，推荐直接填入两者的平均值
    decay_times = np.array([
        0.0500216, 0.4001728, 0.0500216, 0.4001728, 0.1000432, 0.3501512, 0.3001296, 0.3001296, 0.2000864, 0.5502376, 0.4001728, 0.250108, 0.3501512
    ])  # 这是一个示例数组，请用您的实际 13 个数据替换它
    
    T_cut = 2.9012528      # 观测截断时间：3 秒
    bin_width = 0.05 # 时间分辨率：50 ms
    dt = bin_width / 2.0 # 测量半宽度：25 ms

    # ==================== 2. 设置贝叶斯参数空间 ====================
    # 扫描寿命 tau 的范围（从 0.1 秒到 20.0 秒，步长 0.005 秒）
    tau_grid = np.linspace(0.1, 20.0, 4000)
    
    # 使用杰弗里斯无信息先验 (Jeffreys Prior): P(tau) \propto 1/tau
    # 如果您倾向于完全平直的先验，可以将其设为 np.ones_like(tau_grid)
    prior = 1.0 / tau_grid

    # ==================== 3. 计算对数似然函数 ====================
    log_likelihood = np.zeros_like(tau_grid)
    
    for i, tau in enumerate(tau_grid):
        # 引入 50 ms 区间积分修正后的单事件概率
        # P(t_i | tau) = [exp(-(t_i - dt)/tau) - exp(-(t_i + dt)/tau)] / [1 - exp(-T_cut/tau)]
        # 等价于: [exp(-t_i/tau) * 2 * sinh(dt/tau)] / [1 - exp(-T_cut/tau)]
        
        # 避免除以 0 或溢出
        numerator = np.exp(-decay_times / tau) * 2.0 * np.sinh(dt / tau)
        denominator = 1.0 - np.exp(-T_cut / tau)
        
        # 单个事件的条件概率
        p_event = numerator / denominator
        
        # 累加对数似然
        if np.any(p_event <= 0):
            log_likelihood[i] = -np.inf
        else:
            log_likelihood[i] = np.sum(np.log(p_event))

    # ==================== 4. 计算后验概率分布 ====================
    # 为了防止数值溢出，先减去最大对数似然值
    max_log_val = np.max(log_likelihood)
    likelihood = np.exp(log_likelihood - max_log_val)
    
    # 后验概率 = 似然值 * 先验
    posterior_unnorm = likelihood * prior
    
    # 归一化（使曲线下面积为 1）
    area = simps(posterior_unnorm, tau_grid)
    posterior = posterior_unnorm / area

    # ==================== 5. 提取特征值与非对称误差 ====================
    # 1. 最优估计值 (最大后验点 MAP)
    map_idx = np.argmax(posterior)
    tau_best = tau_grid[map_idx]
    
    # 2. 计算 68.3% 最高后验密度区间 (HDI, 对应 1-sigma 误差)
    # 降序排列后验概率，寻找累积面积达到 68.3% 的阈值
    sorted_idx = np.argsort(posterior)[::-1]
    sorted_post = posterior[sorted_idx]
    sorted_grid = tau_grid[sorted_idx]
    
    # 计算积分步长
    d_tau = tau_grid[1] - tau_grid[0]
    cumulative_area = np.cumsum(sorted_post) * d_tau
    
    # 找到包含 68.3% 概率的区域
    hdi_indices = sorted_idx[cumulative_area <= 0.683]
    hdi_grid = tau_grid[hdi_indices]
    
    # 区间边界
    tau_low = np.min(hdi_grid)
    tau_high = np.max(hdi_grid)
    
    # 计算非对称误差
    err_low = tau_best - tau_low
    err_high = tau_high - tau_best

    # ==================== 6. 打印结果 ====================
    print("=" * 50)
    print(" 贝叶斯估算结果 (包含 50ms 时间分辨修正)")
    print("=" * 50)
    print(f"数据点个数 (去重合并后): n = {len(decay_times)}")
    print(f"同核异能态寿命最优值 tau: {tau_best:.3f} s")
    print(f"68.3% 置信区间 (1-sigma 误差): [{tau_low:.3f}, {tau_high:.3f}] s")
    print(f"最终表示形式: tau = {tau_best:.2f} (+{err_high:.2f} / -{err_low:.2f}) s")
    print("=" * 50)

    # ==================== 7. 绘制后验概率曲线图 ====================
    plt.figure(figsize=(9, 6))
    plt.plot(tau_grid, posterior, label="Posterior Probability Density", color='tab:blue', lw=2)
    plt.axvline(tau_best, color='tab:red', linestyle='--', label=f'Best Estimate (MAP) = {tau_best:.2f} (+{err_high:.2f}/-{err_low:.2f}) s')
    plt.fill_between(tau_grid, 0, posterior, 
                     where=(tau_grid >= tau_low) & (tau_grid <= tau_high), 
                     color='tab:red', alpha=0.4, edgecolor=None, label='68.3% HDI (1-sigma range)')
    
    plt.title("Isomer State Lifetime Posterior Distribution", fontsize=14)
    plt.xlabel("Lifetime $\\tau$ (seconds)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.xlim(0.1, max(T_cut, tau_high * 1.5))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.show()

if __name__ == "__main__":
    run_bayesian_estimation()
