""" import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque

def generate_random_metrics(ecs_count, algorithms, base_factors):
    """
"""
    metrics = {
        "direct_hit_rate": {},
        "edge_hit_rate": {},
        "data_weighted_stability": {}
    }

    for metric in metrics:
        for algo in algorithms:
            base = base_factors.get(algo, 1.0)
            metrics[metric][algo] = [round(random.uniform(0.05, 0.2) * base, 3) for _ in range(ecs_count)]

    return metrics

def plot_ecs_metrics(ecs_ids, metrics, algorithms, title_prefix=""):
    """
"""
    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:  # 如果只有一个指标
        axs = [axs]

    for idx, (metric_name, metric_data) in enumerate(metrics.items()):
        ax = axs[idx]
        for algo in algorithms:
            if algo in metric_data:
                ax.plot(ecs_ids, metric_data[algo], label=algo, marker="o")

        ax.set_title(f"{title_prefix} {metric_name.replace('_', ' ').capitalize()}", fontsize=14)
        ax.set_xlabel("ECS", fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').capitalize(), fontsize=12)
        ax.legend()
        ax.grid(linestyle=":")

    plt.tight_layout()
    plt.show()

def plot_global_metrics_over_time(algorithms, time_steps, base_factors):
    """

"""
    plt.figure(figsize=(8, 5))

    for algo in algorithms:
        base = base_factors.get(algo, 1.0)
        global_stability = [round(random.uniform(0.6, 1.0) * base, 3) for _ in range(time_steps)]
        plt.plot(range(1, time_steps + 1), global_stability, label=algo, marker="o")

    plt.title("Global Stability Over Time", fontsize=16)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Global Stability", fontsize=12)
    plt.legend()
    plt.grid(linestyle=":")
    plt.tight_layout()
    plt.show()

def all_in_one_plot(ecs_count=20, time_steps=10, base_factors=None):
    """

"""
    if base_factors is None:
        base_factors = {"FIFO": 1.0, "LRU": 1.0, "GAME": 1.2}

    ecs_ids = [f"ECS{i + 1}" for i in range(ecs_count)]
    algorithms = ["FIFO", "LRU", "GAME"]

    # 生成指标数据
    metrics = generate_random_metrics(ecs_count, algorithms, base_factors)

    # 绘制 ECS 指标对比图
    plot_ecs_metrics(ecs_ids, metrics, algorithms, title_prefix="ECS Metrics")

    # 绘制全局稳定性随时间变化的图像
    plot_global_metrics_over_time(algorithms, time_steps, base_factors)

if __name__ == "__main__":
    all_in_one_plot()
 """
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from collections import deque

def generate_random_metrics(ecs_count, algorithms, base_factors, variance_factor=1.0):
    """
    随机生成 ECS 指标数据，支持控制方差和算法表现放缩。

    参数：
    ecs_count: int, ECS 的数量。
    algorithms: list[str], 算法列表（如 ["FIFO", "LRU", "GAME"]）。
    base_factors: dict[str, float], 各算法的放缩因子，用于调整表现。
    variance_factor: float, 控制随机数据的方差（越小波动越小，越大波动越大）。

    返回：
    dict[str, dict[str, list[float]]]: 各指标数据。
    """
    metrics = {
        "direct_hit_rate": {},
        "edge_hit_rate": {},
        "data_weighted_stability": {}
    }

    for metric in metrics:
        for algo in algorithms:
            middle = 0.7
            base = base_factors.get(algo, 1.0)  # 获取放缩因子
            # 随机生成数据，控制方差
            min_value = middle - 0.1 * variance_factor
            max_value = middle + 0.2 * variance_factor
            raw_values = [random.uniform(min_value, max_value) for _ in range(ecs_count)]
            
            # 按放缩因子调整数据
            if metric != 'direct_hit_rate':
                scaled_values = [min(round(value * base, 3), 0.95) for value in raw_values]
            else:
                scaled_values = raw_values
            metrics[metric][algo] = scaled_values

    return metrics

def plot_ecs_metrics(ecs_ids, metrics, algorithms, title_prefix="", output_dir="plots"):
    """
    绘制每个 ECS 的相关指标对比图。

    参数：
    ecs_ids: list[str], ECS 的 ID 列表（如 ["ECS1", "ECS2", "ECS3"]）。
    metrics: dict[str, dict[str, list[float]]], 各指标的算法数据。
    algorithms: list[str], 算法列表（如 ["FIFO", "LRU", "GAME"]）。
    title_prefix: str, 图表标题的前缀。
    output_dir: str, 输出文件夹。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:  # 如果只有一个指标
        axs = [axs]

    for idx, (metric_name, metric_data) in enumerate(metrics.items()):
        ax = axs[idx]
        for algo in algorithms:
            if algo in metric_data:
                ax.plot(ecs_ids, metric_data[algo], label=algo, marker="o")
        # 设置纵轴范围为 0 到 1
        ax.set_ylim(0, 1)
        ax.set_title(f"{title_prefix} {metric_name.replace('_', ' ').capitalize()}", fontsize=14)
        ax.set_xlabel("ECS", fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').capitalize(), fontsize=12)
        ax.legend()
        ax.grid(linestyle=":")

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{title_prefix.strip().replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_global_metrics_over_time(algorithms, time_steps, base_factors, output_dir="plots"):
    """
    绘制全局稳定性随时间变化的图像。

    参数：
    algorithms: list[str], 算法列表（如 ["FIFO", "LRU", "GAME"]）。
    time_steps: int, 时间步数量。
    base_factors: dict[str, float], 各算法的基准因子，用于调整绝对值。
    output_dir: str, 输出文件夹。

    返回：
    None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(8, 5))

    for algo in algorithms:
        base = base_factors.get(algo, 1.0)
        global_stability = [min(round(random.uniform(0.6, 1.0) * base, 3), 0.95) for _ in range(time_steps)]
        plt.plot(range(1, time_steps + 1), global_stability, label=algo, marker="o")

    plt.title("Global Stability Over Time", fontsize=16)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Global Stability", fontsize=12)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(linestyle=":")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "Global_Stability_Over_Time.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def all_in_one_plot(ecs_count=5, time_steps=10, base_factors=None, output_dir="plots"):
    """
    综合绘图函数，生成 ECS 和全局稳定性指标。

    参数：
    ecs_count: int, ECS 的数量。
    time_steps: int, 时间步数量。
    base_factors: dict[str, float], 各算法的基准因子，用于调整绝对值。
    output_dir: str, 输出文件夹。

    返回：
    None
    """
    if base_factors is None:
        base_factors = {"FIFO": 1.0, "LRU": 1.0, "GAME": 1.2}

    ecs_ids = [f"ECS{i + 1}" for i in range(ecs_count)]
    algorithms = ["FIFO", "LRU", "GAME"]

    # 生成指标数据
    metrics = generate_random_metrics(ecs_count, algorithms, base_factors)

    # 绘制每种算法单独的指标图
    for algo in algorithms:
        algo_metrics = {metric: {algo: data[algo]} for metric, data in metrics.items()}
        plot_ecs_metrics(ecs_ids, algo_metrics, [algo], title_prefix=f"{algo} Metrics", output_dir=output_dir)

    # 绘制所有算法的指标图
    plot_ecs_metrics(ecs_ids, metrics, algorithms, title_prefix="All Algorithms Metrics", output_dir=output_dir)

    # 绘制全局稳定性随时间变化的图像
    plot_global_metrics_over_time(algorithms, time_steps, base_factors, output_dir=output_dir)

if __name__ == "__main__":
    all_in_one_plot()
