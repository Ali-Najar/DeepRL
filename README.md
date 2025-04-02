<p align="center">
  <img src="assets/cover.jpg" alt="DeepRL Logo" width="500"/>
</p>

# 🧠 Deep Reinforcement Learning

This repository contains a series of tasks and Jupyter notebooks that guide you through the fundamentals and advanced topics of Deep Reinforcement Learning (DRL).  
Each part focuses on a specific area, building from foundational concepts to more complex methods.

---

## 📚 Table of Contents

1. 📘 [Introduction to RL](#1-introduction-to-rl)
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

<a name="1-introduction-to-rl"></a>
## 📘 1. Introduction to RL

In this section, we introduce the foundational principles of reinforcement learning through experiments on standard environments and a custom-designed Gym environment. The work includes detailed analyses of hyperparameters, learning curves, and performance metrics.


## 🧾 Contents

- 🌄 [Overview](#overview_01)
- 🧪 [Experiments on Predefined Environments](#experiments-on-predefined-environments_01)
  - ❄️ [FrozenLake-v1](#frozenlake-v1_01)
  - 🎡 [CartPole-v1](#cartpole-v1_01)
  - 📊 [DQN vs PPO: Quick Comparison](#dqn-vs-ppo-comparison_01)
- 🦖 [Custom Environment: Chrome-Dino](#custom-environment-chrome-dino_01)
- 📹 [Media](#media_01)
- 🔗 [References](#references_01)

<a name="overview_01"></a>
## 🌄 Overview

This part of the project establishes the core concepts of Reinforcement Learning (RL) and evaluates two well-known environments—FrozenLake-v1 and CartPole-v1—using both value-based (DQN) and policy-based (PPO) methods. In addition, a custom environment based on the Chrome-Dino game is implemented using Gymnasium and Pygame to showcase the application of RL in interactive settings.

<a name="experiments-on-predefined-environments_01"></a>
## 🧪 Experiments on Predefined Environments

<a name="frozenlake-v1_01"></a>
### ❄️ FrozenLake-v1

In the FrozenLake-v1 environment, multiple models were trained using various hyperparameter settings. Key aspects of this experiment include:

- **Model Variants:**  
  - PPO and DQN were tested with different learning rates, discount factors, varying timesteps, and replay buffer sizes.
  - Additional experiments included the use of reward wrappers to analyze the impact of step costs.

- **Performance Analysis:**  
  - Learning curves were evaluated in terms of rewards per episode and mean episode lengths.
  - The results indicate that PPO generally exhibits smoother convergence, whereas DQN shows greater fluctuation and slower convergence.

<a name="cartpole-v1_01"></a>
### 🎡 CartPole-v1

For the CartPole-v1 environment, the experiments focused on the following:

- **Hyperparameter Tuning:**  
  - Models were trained with different learning rates and discount factors, revealing that lower learning rates lead to more stable convergence.
  - Sufficient training timesteps were critical, with PPO requiring approximately 70,000 timesteps to achieve reliable performance.

- **Observations:**  
  - The continuous state dynamics of CartPole posed additional challenges, particularly for the DQN approach.


<a name="dqn-vs-ppo-comparison_01"></a>
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


<a name="custom-environment-chrome-dino_01"></a>
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


<a name="media_01"></a>
## 📹 Media

Below is the Chrome-Dino test video.

<p align="center">
  <img src="01_Introduction to RL/output.gif" alt="Chrome Dino PPO Agent Demo" width="300"/>
  <br/>
  <em>PPO agent running in the custom Chrome-Dino environment.</em>
</p>

<a name="references_01"></a>
## 🔗 References
- [📄 Chrome-Dino Environment (by MaxRohowsky)](https://github.com/MaxRohowsky/chrome-dinosaur)

---

<a name="2-value-based-methods"></a>
## 📈 2. Value-Based Methods

This section explores value-based reinforcement learning, including epsilon-greedy strategies, n-step methods, and deep Q-learning approaches (DQN vs. DDQN). Experiments focus on learning dynamics, stability, and convergence.

## 🧾 Contents

- 🎲 [Epsilon-Greedy Exploration](#epsilon-greedy_02)
- 🔁 [N-step SARSA and Q-Learning](#n-step_02)
- 🧠 [DQN vs DDQN](#dqn-vs-ddqn_02)
- 🔗 [References](#references_02)

<a name="epsilon-greedy_02"></a>
## 🎲 Epsilon-Greedy Exploration

We analyzed different fixed and decaying $\varepsilon$-values in a CliffWalking environment and evaluated their effect on regret, learning curves, and policies.

### 📊 Regret Behavior (Fixed Epsilon)

<div align="center">

| Epsilon | Behavior |
|--------|----------|
| 0.1 | Fast convergence, minimal long-term regret |
| 0.5 | Occasional jumps from instability |
| 0.9 | Purely exploratory, linear regret growth |

</div>

<p align="center">
  <img src="./02_Value-Based Methods/assets/regret0.1.png" width="300"/>
  <img src="./02_Value-Based Methods/assets/regret0.5.png" width="300"/>
  <img src="./02_Value-Based Methods/assets/regret0.9.png" width="300"/>
  <br/>
  <em>From left to right, regret curves for ε = 0.1, 0.5, and 0.9 respectively.</em>
</p>


### 📊 Epsilon Decay Strategies

We tested fast, medium, and slow epsilon decay and analyzed the rewards and policies near the cliff.

<p align="center">
  <img src="./02_Value-Based Methods/assets/decay.png" width="500"/>
  <br/>
  <em>Comparison of different decay rates.</em>
</p>

<p align="center">
  <img src="./02_Value-Based Methods/assets/fast.png" width="300"/>
  <img src="./02_Value-Based Methods/assets/medium.png" width="300"/>
  <img src="./02_Value-Based Methods/assets/slow.png" width="300"/>
  <br/>
  <em>From left to right, policies for fast, medium and slow decay respectively.</em>
</p>

>  **Observation:** Fast decay leads to quicker convergence but riskier policies. Slower decay enforces caution near the cliff.

<a name="n-step_02"></a>
## 🔁 N-step SARSA and Q-Learning

We compared performance using different values of n (1, 2, 5) for both SARSA and Q-learning.

### 🔄 Q-learning

<p align="center">
  <img src="./02_Value-Based Methods/assets/q-n1.png" width="300"/>
  <img src="./02_Value-Based Methods/assets/q-n2.png" width="300"/>
  <img src="./02_Value-Based Methods/assets/q-n5.png" width="300"/>
  <br/>
  <em>Q-Learning with n = 1, 2, 5.</em>
</p>

### 🔄 SARSA

<p align="center">
  <img src="./02_Value-Based Methods/assets/sarsa-n1.png" width="300"/>
  <img src="./02_Value-Based Methods/assets/sarsa-n2.png" width="300"/>
  <img src="./02_Value-Based Methods/assets/sarsa-n5.png" width="300"/>
  <br/>
  <em>SARSA with n = 1, 2, 5.</em>
</p>


<a name="dqn-vs-ddqn_02"></a>
## 🧠 DQN vs DDQN

We implemented both DQN and DDQN on CartPole and compared their stability, value estimates, and learning curves.

### 📈 Performance Comparison

<p align="center">
  <img src="./02_Value-Based Methods/assets/ddqn-dqn.png" width="400"/>
  <br/>
  <em>DDQN shows better stability and higher average reward per episode.</em>
</p>

### 📈 Value Estimates

<p align="center">
  <img src="./02_Value-Based Methods/assets/ddqn-val.png" width="350"/>
  <img src="./02_Value-Based Methods/assets/dqn-val.png" width="350"/>
  <br/>
  <em>DQN overestimates value consistently compared to DDQN. (Left plot DDQN, right plot DQN)</em>
</p>


### 🔧 Improvements Explored

- 🧠 **Maxmin Q-learning**: Uses ensembles to mitigate overestimation bias.
- ⚖️ **Prioritized Experience Replay (PER)**: Samples transitions based on TD-error for improved learning efficiency.


<a name="references_02"></a>
## 🔗 References

- [📄 Sutton & Barto – Reinforcement Learning: An Introduction (2nd Ed)](http://incompleteideas.net/book/the-book-2nd.html)
- [📄 Grokking Deep Reinforcement Learning](https://www.manning.com/books/grokking-deep-reinforcement-learning)
- [📄 Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [📄 Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [📄 Maxmin Q-learning](https://arxiv.org/abs/2002.06487)

---

<a name="3-policy-based-methods"></a>
# 🎯 3. Policy-Based Methods

This part explores policy gradient techniques through a series of experiments. It includes a comparison between REINFORCE and Genetic Algorithms, performance analysis of baselines in REINFORCE, training in continuous action spaces, and drawbacks of policy gradients.


## 🧾 Contents

- 🧬 [REINFORCE vs Genetic Algorithm](#reinforce-vs-ga_03)
- ⚖️ [REINFORCE: Baseline vs No Baseline](#baseline-comparison_03)
- 🌄 [REINFORCE in Continuous Action Space](#reinforce-continuous_03)
- ❌ [Policy Gradient Drawbacks](#pg-drawbacks_03)
- 🔗 [References](#references_03)


<a name="reinforce-vs-ga_03"></a>
## 🧬 REINFORCE vs Genetic Algorithm

We trained both REINFORCE and GA on a custom 7×7 GridWorld with penalties and a goal reward. GA uses mutation/crossover, while REINFORCE optimizes action probabilities via gradients.

<div align="center">


| Aspect                | REINFORCE       | Genetic Algorithm |
|-----------------------|------------------|-------------------|
| Learning Type         | Gradient-based    | Evolution-based   |
| Sample Efficiency     | High              | Low               |
| Exploration Method    | Stochastic policy | Mutation/crossover|
| Convergence Speed     | Faster            | Slower            |
| Stability             | Lower             | Higher            |

</div>

> 💡 REINFORCE converges faster but is more volatile. GA is slower, yet more stable due to population-based learning.

<a name="baseline-comparison_03"></a>
## ⚖️ REINFORCE: Baseline vs No Baseline

We trained REINFORCE with and without a value baseline on CartPole-v1.

<p align="center">
  <img src="./03_Policy-Based Methods/assets/baseline.png" width="600"/>
  <br/>
  <em>Baseline significantly stabilizes learning.</em>
</p>

- **Without baseline**: High variance, unstable learning, sharp jumps in rewards.
- **With baseline**: Smoother reward curve, more consistent convergence.

> 📌 A value baseline (typically \( V(s_t) \)) helps reduce variance in policy gradients, improving learning stability.

<a name="reinforce-continuous_03"></a>
## 🌄 REINFORCE in Continuous Action Space

We trained a Gaussian policy with REINFORCE on MountainCarContinuous-v0.

<p align="center">
  <img src="./03_Policy-Based Methods/assets/mountainCar.png" width="600"/>
  <br/>
  <em>Episode reward curve for REINFORCE in continuous setting</em>
</p>

**Observation space**:  
- Position \([-1.2, 0.6]\)  
- Velocity \([-0.07, 0.07]\)

**Action space**:  
- Continuous force \([-1.0, 1.0]\)

> ✅ Optimal policy uses momentum and gravity to minimize energy while reaching the goal.

We used independent `nn.Parameter` for standard deviation in the Gaussian policy to avoid coupling it to the state.


### 💡 Preventing Catastrophic Forgetting

We applied several strategies:
- 🌀 **Experience Replay** – reuse older transitions.
- 🧊 **Target Networks** – stabilize bootstrapped value estimates.
- 📐 **Adaptive Learning Rates** – via optimizers like Adam.
- 🔀 **Entropy Regularization** – keeps policy exploratory early in training.


<a name="pg-drawbacks_03"></a>
## ❌ Policy Gradient Drawbacks

Despite being powerful, policy gradients suffer from:

<div align="center">

| Challenge               | Explanation                                                                 |
|------------------------|------------------------------------------------------------------------------|
| High Variance          | Due to stochastic rewards and sampling entire episodes.                     |
| Sample Inefficiency    | On-policy updates require lots of interactions.                             |
| Slow Convergence       | Sensitive to learning rate, gradient noise.                                 |
| Credit Assignment      | Hard to attribute rewards to early actions.                                 |
| No Off-Policy Learning | Cannot leverage past data like DQN.                                         |

</div>

### 🧠 FrozenLake: DQN vs REINFORCE

<p align="center">
  <img src="./03_Policy-Based Methods/assets/dqn.png" width="300"/>
  <img src="./03_Policy-Based Methods/assets/reinforce.png" width="300"/>
  <br/>
  <em>DQN (left) succeeds; REINFORCE (right) fails to learn.</em>
</p>

- **DQN** performs far better in sparse reward and discrete action environments.
- **REINFORCE** struggles due to high variance and lack of replay or off-policy learning.


<a name="references_03"></a>
## 🔗 References

- [📘 Policy Gradient Lecture – Amirreza Farahmand](https://amfarahmand.github.io/IntroRL/lectures/lec06.pdf)
- [🧱 CartPole Environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)
- [🚗 MountainCarContinuous Environment](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)
- [❄️ FrozenLake Environment](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/)

---

