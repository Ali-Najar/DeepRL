# 🧠 Deep Reinforcement Learning

This repository contains a series of tasks and Jupyter notebooks that guide you through the fundamentals and advanced topics of Deep Reinforcement Learning (DRL).  
Each part focuses on a specific area, building from foundational concepts to more complex methods.

---

## 📚 Table of Contents

1.  [📘 Introduction to RL](#1-introduction-to-rl)
2. 📈 [Value-Based Methods](#2-value-based-methods)
3. 🎯 [Policy-Based Methods](#3-policy-based-methods)
4. 🧩 [Advanced Methods](#4-advanced-methods)
5. 🧠 [Model-Based Methods](#5-model-based-methods)
6. 🎰 [Multi-Armed Bandits](#6-multi-armed-bandits)
7. 🔢 [Value-Based Theory](#7-value-based-theory)
8. 🧭 [Policy-Based Theory](#8-policy-based-theory)
9. 🧬 [Advanced Theory](#9-advanced-theory)
10. 🕵️‍♂️ [Exploration Methods](#10-exploration-methods)
11. 👥 [Imitation & Inverse RL](#11-imitation-and-inverse-rl)
12. 📦 [Offline Methods](#12-offline-methods)
13. 🤖 [Multi-Agent Methods](#13-multi-agent-methods)
14. 🏗️ [Hierarchical & Meta RL](#14-hierarchical-and-meta-rl)

---

## 📘 1. Introduction to RL

In this section, we introduce the foundational principles of reinforcement learning through experiments on standard environments and a custom-designed Gym environment. The work includes detailed analyses of hyperparameters, learning curves, and performance metrics.


## 🧾 Contents

- 🌄 [Overview](#overview)
- 🧪 [Experiments on Predefined Environments](#experiments-on-predefined-environments)
  - ❄️ [FrozenLake-v1](#frozenlake-v1)
  - 🎡 [CartPole-v1](#cartpole-v1)
  - 📊 [DQN vs PPO: Quick Comparison](#dqn-vs-ppo-comparison)
- 🦖 [Custom Environment: Chrome-Dino](#custom-environment-chrome-dino)
- 📹 [Media](#media)
- 🔗 [References](#references)

## 🌄 Overview

This part of the project establishes the core concepts of Reinforcement Learning (RL) and evaluates two well-known environments—FrozenLake-v1 and CartPole-v1—using both value-based (DQN) and policy-based (PPO) methods. In addition, a custom environment based on the Chrome-Dino game is implemented using Gymnasium and Pygame to showcase the application of RL in interactive settings.

## 🧪 Experiments on Predefined Environments

### ❄️ FrozenLake-v1

In the FrozenLake-v1 environment, multiple models were trained using various hyperparameter settings. Key aspects of this experiment include:

- **Model Variants:**  
  - PPO and DQN were tested with different learning rates, discount factors, varying timesteps, and replay buffer sizes.
  - Additional experiments included the use of reward wrappers to analyze the impact of step costs.

- **Performance Analysis:**  
  - Learning curves were evaluated in terms of rewards per episode and mean episode lengths.
  - The results indicate that PPO generally exhibits smoother convergence, whereas DQN shows greater fluctuation and slower convergence.

### 🎡 CartPole-v1

For the CartPole-v1 environment, the experiments focused on the following:

- **Hyperparameter Tuning:**  
  - Models were trained with different learning rates and discount factors, revealing that lower learning rates lead to more stable convergence.
  - Sufficient training timesteps were critical, with PPO requiring approximately 70,000 timesteps to achieve reliable performance.

- **Observations:**  
  - The continuous state dynamics of CartPole posed additional challenges, particularly for the DQN approach.
 
### 📊 DQN vs PPO: Quick Comparison

<div align="center">
  
| Environment     | Algorithm | Convergence Speed | Stability | Final Performance |
|----------------|-----------|-------------------|-----------|-------------------|
| FrozenLake-v1   | DQN       | Slow              | Low       | Medium            |
|                 | PPO       | Fast              | High      | High              |
| CartPole-v1     | DQN       | Medium            | Moderate  | High              |
|                 | PPO       | Fast              | High      | High              |

</div>

<p align="center">
  <img src="./01_Introduction to RL/frozen-lake.png" alt="DQN vs PPO on FrozenLake-v1" width="600"/>
  <br/>
  <em>Comparison of DQN (🟣) and PPO (⚫) on FrozenLake-v1.</em>
</p>

<p align="center">
  <img src="./01_Introduction to RL/cartpole.png" alt="DQN vs PPO on CartPole-v1" width="300"/>
  <br/>
  <em>Comparison of DQN (🟣) and PPO (⚫) on CartPole-v1.</em>
</p>

## 🦖 Custom Environment: Chrome-Dino

A custom Gymnasium environment was developed to simulate the Chrome-Dino game using Pygame. This environment is defined by:

- **State Space:**  
  - A 3-dimensional vector including:
    - `dino_y`: Vertical position (range: 0–400)
    - `dino_vel_y`: Vertical velocity (range: –20 to 20)
    - `closest_dist`: Distance to the nearest obstacle (range: 0–1000)

- **Action Space:**  
  - Discrete actions: `0` (No Operation), `1` (Jump), and `2` (Duck).

- **Reward Function:**  
  - +1 per timestep for survival,
  - +5 for successfully passing an obstacle,
  - -200 upon collision (which terminates the episode).

The agent was trained using PPO with an MLP policy over 2,000,000 timesteps. This setup demonstrates how reinforcement learning techniques can be applied to interactive environments with dynamic visual feedback.

## 📹 Media

Below is the Chrome-Dino test video.

<p align="center">
  <img src="01_Introduction to RL/output.gif" alt="Chrome Dino PPO Agent Demo" width="300"/>
  <br/>
  <em>PPO agent running in the custom Chrome-Dino environment.</em>
</p>
