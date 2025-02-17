import time
import gym
import numpy as np
import pygame



def render_rotated_mirrored_frame(env, screen, sleep_time):
    frame = env.render(mode='rgb_array')
    frame = np.array(frame)

    # 将图像旋转90度（顺时针），然后进行镜像对调
    rotated_frame = np.rot90(frame, 1)  # 旋转90度（顺时针）
    mirrored_frame = np.flip(rotated_frame, axis=0)  # 上下镜像对调

    # 将 NumPy 数组转换为 pygame Surface
    frame_surface = pygame.surfarray.make_surface(mirrored_frame)

    # 在屏幕上显示图像
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()
    time.sleep(sleep_time)

# ----------------------------
# 初始化环境和 pygame
# ----------------------------
# 创建 FrozenLake 环境（关闭滑动，使学习更稳定，使用 render_mode='rgb_array' 以便获取图像数据）
env = gym.make("FrozenLake-v1", is_slippery=False) #, render_mode="rgb_array")

# 重置环境以获得初始图像
init_state = env.reset()
# 兼容 gym 0.26+（返回 (state, info)）或旧版本
if isinstance(init_state, tuple):
    state = init_state[0]
else:
    state = init_state

frame = env.render(mode='rgb_array') # 获取初始帧，格式为 (height, width, 3)

print(f"frame={frame}")
height, width, _ = frame.shape

# 初始化 pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("FrozenLake Q-Learning Training with pygame")

# ----------------------------
# Q-Learning 参数设置
# ----------------------------
n_states = env.observation_space.n  # 状态数（例如 16）
n_actions = env.action_space.n  # 动作数（例如 4）
goal_state = n_states - 1  # 终点状态

# 初始化 Q 表
Q = np.zeros((n_states, n_actions))

# 超参数
alpha = 0.1  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 100  # 训练轮数（你可以根据需要增加轮数）

# ----------------------------
# 训练 Q-Learning 并使用 pygame 可视化训练过程
# ----------------------------
print("开始训练……")
for episode in range(num_episodes):
    # 重置环境
    reset_return = env.reset()
    if isinstance(reset_return, tuple):
        state = reset_return[0]
    else:
        state = reset_return
    done = False

    while not done:
        # 处理 pygame 事件（例如关闭窗口）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # ε-贪心策略选择动作
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # 执行动作，获取反馈
        step_return = env.step(action)
        # 根据 gym 版本的不同，step 返回的值可能不同（4 个或 5 个），此处按返回 4 个值处理
        if len(step_return) == 4:
            next_state, reward, done, info = step_return
        else:
            next_state, reward, done, truncated, info = step_return

        # 自定义奖励策略（可根据需要调整）
        if next_state == state:
            reward = -0.01
        elif done and next_state != goal_state:
            reward = -0.1

        # 更新 Q 表（Bellman 更新公式）
        best_next_action = np.argmax(Q[next_state, :])
        Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
        state = next_state

        #render_rotated_mirrored_frame(env, screen,0.1)

        # 添加短暂停顿，便于观察
        #time.sleep(0.1)

    # 每个 episode 结束后降低探索率
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode + 1}/{num_episodes} 完成.")

print("🎉 训练完成！")

# ----------------------------
# 利用 pygame 可视化训练后智能体的最优路径
# ----------------------------
print("展示训练后智能体执行最优策略……")
reset_return = env.reset()
if isinstance(reset_return, tuple):
    state = reset_return[0]
else:
    state = reset_return
done = False

render_rotated_mirrored_frame(env, screen,0.5)
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    # 获取并显示当前环境帧
    frame = env.render(mode='rgb_array')
    frame = np.array(frame)

    # 根据 Q 表选择最优动作
    action = np.argmax(Q[state, :])
    state, reward, done, info = env.step(action)
    render_rotated_mirrored_frame(env, screen,0.5)



# 保持窗口打开，等待用户关闭
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()