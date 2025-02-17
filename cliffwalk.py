import time
import gym
import numpy as np
import pygame


def render_rotated_mirrored_frame(env, screen, sleep_time):
    """
    获取环境图像并进行旋转+镜像处理，然后在 pygame 屏幕上显示。

    :param env: Gym 环境对象
    :param screen: Pygame 屏幕对象
    :param sleep_time: 每帧显示的时间间隔（秒）
    """
    frame = env.render(mode='rgb_array')
    frame = np.array(frame)

    # 旋转90度（顺时针），然后进行镜像对调
    rotated_frame = np.rot90(frame, 1)
    mirrored_frame = np.flip(rotated_frame, axis=0)

    # 转换为 pygame Surface 并显示
    frame_surface = pygame.surfarray.make_surface(mirrored_frame)
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()
    time.sleep(sleep_time)


# ----------------------------
# 初始化环境和 pygame
# ----------------------------

from gym.envs.registration import register


env = gym.make("CliffWalking-v0")# render_mode="rgb_array")

# 获取状态空间大小
n_states = env.observation_space.n
n_actions = env.action_space.n
goal_state = n_states - 1  # 终点状态

# 获取环境的初始图像
init_state = env.reset()
if isinstance(init_state, tuple):
    state = init_state[0]
else:
    state = init_state

frame = env.render(mode='rgb_array')

height, width, _ = frame.shape

# 初始化 pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("CliffWalking Q-Learning Training with pygame")

# ----------------------------
# Q-Learning 参数设置
# ----------------------------
Q = np.zeros((n_states, n_actions))  # 初始化 Q 表
alpha = 0.1  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 10  # 训练轮数

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
        if len(step_return) == 4:
            next_state, reward, done, info = step_return
        else:
            next_state, reward, done, truncated, info = step_return

        # 自定义 Cliff Walking 的奖励策略（原始奖励已经适合 Q-Learning）
        if done and next_state != goal_state:  # 掉落悬崖
            reward = -100

        # 更新 Q 表
        best_next_action = np.argmax(Q[next_state, :])
        Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
        state = next_state

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

render_rotated_mirrored_frame(env, screen, 0.5)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # 获取并显示当前环境帧
    frame = env.render()
    frame = np.array(frame)

    # 选择 Q 表中的最佳动作
    action = np.argmax(Q[state, :])
    state, reward, done, info = env.step(action)

    render_rotated_mirrored_frame(env, screen, 0.5)

# 保持窗口打开，等待用户关闭
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
