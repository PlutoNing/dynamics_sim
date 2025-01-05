import matplotlib.pyplot as plt
import numpy as np
import random,os
from loader import read_csv,read_content_attr

num_servers = 10  # 服务器数量
num_contents = 20  # 内容数量
R = np.random.rand(num_servers)  # 服务器稳定性评分
I = np.random.rand(num_contents)  # 内容重要性评分
connections = [np.random.choice(num_servers, 3, replace=False) for _ in range(num_servers)]  # 每个服务器的连接

# 初始化策略概率
P = np.random.rand(num_servers, num_contents)  # 每个服务器缓存内容的概率
trajectories = []
servers = {i: {"contents": [], "remaining_capacity": 1000} for i in range(20)}

# ecs_ids = ["ECS1", "ECS2", "ECS3"] 扩展容量为20个，一直到ECS20
ecs_ids = [f"ECS{i+1}" for i in range(20)]
direct_hit = [0] * 20
direct_requests = [0] * 20

data_weighted_stability = [0] * 20
global_data_weighted_stability = []

content_sizes, content_importance = read_content_attr()
# 计算总的稳定性，就是centen important的累加
total_importance = sum(content_importance.values())
for content_id in range(2000):
    content_id += 1
    assigned = False
    while not assigned:
        server_id = random.choice(list(servers.keys()))
        size = content_sizes[content_id]
        if servers[server_id]["remaining_capacity"] >= size:
            servers[server_id]["contents"].append(content_id)
            servers[server_id]["remaining_capacity"] -= size
            assigned = True
user_to_server = {user_id+1 : user_id+1  % 20 for user_id in range(200)}
reqs = read_csv(r"C:\Users\14459\PycharmProjects\dynamics_sim\data\ratings.csv", 200)
# 生成一个代表20个服务器健康评分的列表， 值在0.5到1之间
current_health = [round(random.uniform(0.5, 1.0), 2) for _ in range(20)]

def calculate_global_objective(current_health, servers):
    """
    计算全局目标值，即每台服务器的健康评分与承载内容重要性的加权和。

    参数：
    current_health: list[float], 当前20台服务器的健康评分。
    servers: dict, 每台服务器的内容和状态。

    返回：
    float: 全局目标值。
    """
    global_objective = 0
    for server_id, server_data in servers.items():
        server_health = current_health[server_id]
        total_importance = sum(server_data["contents"])
        global_objective += server_health * total_importance
    return global_objective


def update_server_health(current_health):
    """
    随机选择一定比例（0到20%）的服务器，更新其健康评分。

    参数：
    current_health: list[float], 当前20台服务器的健康评分列表。

    返回：
    list[float]: 更新后的健康评分列表。
    """
    # 确保输入为20个健康评分
    assert len(current_health) == 20, "当前健康评分列表的长度必须为20"

    # 随机选择 0 到 20% 的服务器进行更新
    max_to_update = len(current_health) // 5  # 20% 的服务器数量
    num_to_update = random.randint(1, max_to_update)  # 随机选择 1 到 max_to_update 台服务器

    to_update_indices = random.sample(range(len(current_health)), num_to_update)

    # 更新这些服务器的健康评分
    updated_health = current_health[:]
    for index in to_update_indices:
        updated_health[index] = round(random.uniform(0.5, 1.0), 2)  # 新健康评分范围在0.5到1.0之间

    return updated_health
# 动态复制方程
# f(x, y): 当前服务器的复制动态方程
# g(x, y): 另一方服务器的复制动态方程


def f(x, y):
    return x * (1 - x) * (5 * y - 2.5)

def g(x, y):
    return y * (1 - y) * (4 * x - 2)

def payoff_matrix(popularity, discount_factor):
    """
    定义收益矩阵，基于内容重要程度（流行度）和折扣因子。

    参数：
    popularity: float, 内容的重要程度。
    discount_factor: float, 折扣因子。

    返回：
    list[list[float]]: 收益矩阵。
    """
    a1 = popularity * (1 - discount_factor)
    a2 = popularity
    a3 = popularity * discount_factor
    a4 = 0
    b1 = popularity * discount_factor
    b2 = popularity * (1 - discount_factor)
    b3 = popularity
    b4 = 0

    return [[a1, b1], [a2, b2], [a3, b3], [a4, b4]]
def calculate_replication_dynamics(x, y, matrix):
    """
    根据收益矩阵计算复制动态方程。

    参数：
    x: float, 玩家 A 的策略概率。
    y: float, 玩家 B 的策略概率。
    matrix: list[list[float]], 收益矩阵。

    返回：
    tuple[float, float]: dx/dt 和 dy/dt。
    """
    a1, b1 = matrix[0]
    a2, b2 = matrix[1]
    a3, b3 = matrix[2]
    a4, b4 = matrix[3]

    Ex = y * a1 + (1 - y) * a2
    E_x = y * a3 + (1 - y) * a4
    Ey = x * b1 + (1 - x) * b3
    E_y = x * b2 + (1 - x) * b4

    dx_dt = x * (1 - x) * (Ex - E_x)
    dy_dt = y * (1 - y) * (Ey - E_y)

    return dx_dt, dy_dt

def calculateValue_orig(initX, initY, dt, epoch):
    x = []
    y = []
    
    #演化初始赋值
    x.append(initX)
    y.append(initY)
    
    #微量计算及append
    for index in range(epoch):
        tempx = x[-1] + (f(x[-1],y[-1])) * dt
        tempy = y[-1] + (g(x[-1],y[-1])) * dt
 
        x.append(tempx)
        y.append(tempy)

def calculateValue(initX, initY, dt, epoch, matrix):
    """
    计算演化路径。

    参数：
    initX: float, 玩家 A 的初始策略概率。
    initY: float, 玩家 B 的初始策略概率。
    dt: float, 时间步长。
    epoch: int, 演化迭代次数。
    matrix: list[list[float]], 收益矩阵。

    返回：
    tuple[list, list]: x 和 y 的演化路径。
    """
    x = [initX]
    y = [initY]

    for _ in range(epoch):
        dx, dy = calculate_replication_dynamics(x[-1], y[-1], matrix)
        tempx = x[-1] + dx * dt
        tempy = y[-1] + dy * dt
        # 确保策略概率在 [0, 1] 范围内
        tempx = max(0, min(1, tempx))
        tempy = max(0, min(1, tempy))
        x.append(tempx)
        y.append(tempy)

    return (x, y)


def process_requests_lru(current_health, requests, user_to_server_mapping, content_sizes, content_importance):
    """
    """
    servers = {i: {"contents": {}, "remaining_capacity": 1000, "access_order": deque()} for i in range(20)}

    for idx, (user_id, content_id) in enumerate(requests):
        print(f"Processing request {idx + 1}/{len(requests)}")
        server_id = user_to_server_mapping[user_id]
        content_size = content_sizes.get(content_id, 1)
        content_import = content_importance.get(content_id, 1)

        if content_id in servers[server_id]["contents"]:
            print(f"Request handled: User {user_id} requested Content {content_id} -> Served from Server {server_id}")
            # 更新访问顺序
            servers[server_id]["access_order"].remove(content_id)
            servers[server_id]["access_order"].append(content_id)
        else:
            print(f"Request handled: User {user_id} requested Content {content_id} -> Not in Server {server_id}, caching...")
            if servers[server_id]["remaining_capacity"] >= content_size:
                servers[server_id]["contents"][content_id] = content_import
                servers[server_id]["access_order"].append(content_id)
                servers[server_id]["remaining_capacity"] -= content_size
            else:
                # LRU 替换策略
                to_remove = servers[server_id]["access_order"].popleft()
                servers[server_id]["remaining_capacity"] += content_sizes[to_remove]
                del servers[server_id]["contents"][to_remove]

                servers[server_id]["contents"][content_id] = content_import
                servers[server_id]["access_order"].append(content_id)
                servers[server_id]["remaining_capacity"] -= content_size

    return servers

def replicator_dynamics(prob_cache, server_health, importance, cost):
    """
    复制动态方程更新策略概率。

    参数：
    prob_cache: float, 当前缓存策略的概率。
    server_health: float, 服务器健康评分。
    importance: float, 内容重要性。
    cost: float, 缓存成本。

    返回：
    float: 更新后的缓存策略概率。
    """
    payoff_cache = server_health * importance - cost
    payoff_no_cache = 0  # 不缓存的收益假设为0

    avg_payoff = prob_cache * payoff_cache + (1 - prob_cache) * payoff_no_cache

    # 复制动态方程
    return prob_cache + prob_cache * (payoff_cache - avg_payoff)
def simulate_game(current_health, requests, num_rounds=10, convergence_threshold=1e-3):
    """
    模拟服务器博弈的过程，每一轮根据健康评分处理请求。

    参数：
    current_health: list[float], 初始20台服务器的健康评分列表。
    requests: list[tuple], 用户请求列表，每个元组为(user_id, content_id)。
    num_rounds: int, 模拟的博弈轮数。
    convergence_threshold: float, 收敛判断的阈值。

    返回：
    None
    """
    user_to_server = {user_id: user_id % 20 for user_id in range(200)}  # 用户到服务器的映射

    previous_global_objective = 0

    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}:")

        # 更新健康评分
        updated_health = update_server_health(current_health)

        # 处理请求
        servers = process_requests(updated_health, requests, user_to_server, content_sizes, content_importance)

        # 计算全局目标值
        global_objective = calculate_global_objective(updated_health, servers)
        print(f"Global Objective: {global_objective:.4f}")

        # 判断是否收敛
        if abs(global_objective - previous_global_objective) < convergence_threshold:
            print("Converged!")
            break

        previous_global_objective = global_objective

        # 更新当前健康评分
        current_health = updated_health
        print("\n")
def print_all():
    print("Direct Hit Counts:")
    print(direct_hit)
    print("\nDirect Requests:")
    print(direct_requests)
    for server_id, hit_rate in enumerate(direct_hit):
        total_requests = direct_requests[server_id]
        if total_requests > 0:
            print(f"Server {server_id}: {hit_rate / total_requests:.4f}")
        else:
            print(f"Server {server_id}: No requests")

    print("\nData-Weighted Stability:")
    for server_id, stability in enumerate(data_weighted_stability):
        print(f"Server {server_id}: {stability:.4f}")

    print("\nGlobal Data-Weighted Stability:")
    if global_data_weighted_stability:
        print(f"{sum(global_data_weighted_stability) / len(global_data_weighted_stability):.4f}")
    else:
        print("No global data available")

def test_plot_ecs_metrics():
    ecs_ids = ["ECS1", "ECS2", "ECS3"]
    metrics = {
    "direct_hit_rate": {
        "FIFO": [0.1, 0.15, 0.2],
        "LRU": [0.12, 0.18, 0.25],
        "EG-CPS": [0.2, 0.22, 0.3]
    },
    "edge_hit_rate": {
        "FIFO": [0.3, 0.35, 0.4],
        "LRU": [0.28, 0.34, 0.39],
        "EG-CPS": [0.4, 0.45, 0.5]
    },
    "data_weighted_stability": {
        "FIFO": [0.8, 0.82, 0.85],
        "LRU": [0.78, 0.81, 0.83],
        "EG-CPS": [0.85, 0.88, 0.9]
    }
}
    algorithms = ["FIFO", "LRU", "EG-CPS"]
    plot_ecs_metrics(ecs_ids, metrics, algorithms, title_prefix="ECS Metrics")

def plot_ecs_metrics(ecs_ids, metrics, algorithms, title_prefix=""):
    """
    绘制每个 ECS 的相关指标对比图。

    参数：
    ecs_ids: list[str], ECS 的 ID 列表（如 ["ECS1", "ECS2", "ECS3"]）。
    metrics: dict[str, dict[str, list[float]]], 各指标的算法数据。
        示例:
        {
            "direct_hit_rate": {
                "FIFO": [0.1, 0.2, 0.3],
                "LRU": [0.15, 0.25, 0.35],
                ...
            },
            "edge_hit_rate": {
                "FIFO": [0.3, 0.4, 0.5],
                ...
            },
            "data_weighted_stability": {
                "FIFO": [0.8, 0.85, 0.9],
                ...
            },
        }
    algorithms: list[str], 算法列表（如 ["FIFO", "LRU", "EG-CPS"]）。
    title_prefix: str, 图表标题的前缀。
    """
    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))

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

def phase_diagram():
    plt.figure(figsize=(8, 8))
    step = 100
    sampled_trajectories = trajectories[::step]
    
    for n, m in sampled_trajectories:
        plt.plot(n, m, alpha=0.6)

    # 图形参数
    plt.title("Two-Party Evolutionary Game", fontsize=18)
    plt.xlabel("Probability of keeping (x)", fontsize=14)
    plt.ylabel("Probability of giving (y)", fontsize=14)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.grid(linestyle=":", linewidth=0.7, alpha=0.7)
    plt.savefig("two_party_evolutionary_game.png", dpi=300, bbox_inches="tight")
    plt.show()

# 主函数
def main():
    # 初始化
    trajectories = []
    num_paths = 50  # 随机生成50条初始路径
    dt = 0.01       # 步长
    epoch = 500     # 演化迭代次数

    # 随机生成初始点并计算演化路径
    for _ in range(num_paths):
        random_x = random.uniform(0, 1)  # 随机初始化 x
        random_y = random.uniform(0, 1)  # 随机初始化 y
        path = calculateValue_orig(random_x, random_y, dt, epoch)
        trajectories.append(path)
    
    # 绘制相位图
    plt.figure(figsize=(8, 8))
    for n, m in trajectories:
        plt.plot(n, m, alpha=0.6)

    # 图形参数
    plt.title("Two-Party Evolutionary Game", fontsize=18)
    plt.xlabel("Probability of keeping (x)", fontsize=14)
    plt.ylabel("Probability of giving (y)", fontsize=14)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.grid(linestyle=":", linewidth=0.7, alpha=0.7)
    plt.savefig("two_party_evolutionary_game.png", dpi=300, bbox_inches="tight")
    plt.show()

# 执行主函数
if __name__ == "__main__":
    # main()
    print("Hello, world!")
    simulate_game(current_health, reqs, num_rounds=10)
    # phase_diagram()
    # test_plot_ecs_metrics()
    print_all()