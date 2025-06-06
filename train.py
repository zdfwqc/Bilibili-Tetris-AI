import numpy as np
from game_env import GameEnvironment
from dqn_model import DQNAgent
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
from collections import deque
import time
import torch

def plot_rewards(rewards, avg_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward')
    plt.plot(avg_rewards, label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('images/training_rewards.png')
    plt.close()

def train(env, agent, num_episodes=100000, target_update=10, save_interval=200):
    rewards_history = []
    avg_rewards = deque(maxlen=100)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            episode_reward += reward
            
        if episode % target_update == 0:
            agent.update_target_network()
            
        rewards_history.append(episode_reward)
        avg_rewards.append(episode_reward)
        avg_reward = np.mean(avg_rewards)
        
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Epsilon: {agent.epsilon:.2f}")
        print("-" * 50)
        
        if (episode + 1) % save_interval == 0:
            agent.save(f"dqn_model_episode.pth")
            plot_rewards(rewards_history, list(avg_rewards))
            
        time.sleep(0.1)  # 添加小延迟以避免过快执行

if __name__ == "__main__":
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    state_shape = (480, 640)  # 与环境中图像预处理后的尺寸相匹配
    n_actions = 5
    agent = DQNAgent(state_shape, n_actions, device=device)
    # 尝试加载已有模型参数继续训练
    try:
        agent.load("dqn_model_episode.pth")
        print("成功加载已有模型参数")
    except:
        print("未找到已有模型,将从头开始训练")
    
    # 初始化环境
    env = GameEnvironment()
    env.reset()
    # 初始化智能体
    
    print('start train')
    # 开始训练
    train(env, agent) 