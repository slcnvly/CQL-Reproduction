# Mitigating Overestimation Bias in Offline RL: A Comparative Analysis of SAC and CQL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/slcnvly/CQL-Reproduction/blob/master/CQL.ipynb)

This repository provides an implementation and comparative analysis of **Standard Offline Actor-Critic (SAC)** and **Conservative Q-Learning (CQL)**. The objective of this project is to empirically demonstrate the "Overestimation Bias" inherent in standard off-policy algorithms when applied to static datasets, and how CQL's mathematical formulation effectively mitigates this issue to establish a safe lower bound for the policy.



## 1. Introduction: The Extrapolation Error in Offline RL
In Offline Reinforcement Learning, the agent must learn strictly from a fixed dataset without further environmental interaction. Standard off-policy algorithms (like SAC or DDPG) suffer from **Overestimation Bias** in this setting. 

When the agent queries the Q-network for actions that are not present in the dataset (Out-of-Distribution, OOD actions), the Q-network often outputs erroneously high values. The Bellman update bootstraps these false high values, leading to catastrophic policy degradation.

**Conservative Q-Learning (CQL)** addresses this by explicitly penalizing the Q-values of OOD actions during training, ensuring the expected value of a policy under the learned Q-function lower-bounds its true value.



## 2. Methodology & Core Logic
The primary difference between standard SAC and CQL lies in the Critic's loss function. CQL adds a regularization term to the standard Bellman Error.


### Mathematical Formulation (CQL Equation 2)
The objective function for the CQL Q-network is defined as:


$$
\min_{Q} \alpha \mathbb{E}_{s \sim D} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim D}[Q(s, a)] \right] + \frac{1}{2} \mathbb{E}_{s, a, s' \sim D} \left[ (Q(s, a) - \hat{\mathcal{B}}^{\pi}Q(s, a))^2 \right]
$$


* **Push Down:** $\log \sum_a \exp(Q(s, a))$ heavily penalizes (reduces) the Q-values of uniformly sampled, potentially OOD actions.
* **Push Up:** $\mathbb{E}_{a \sim D}[Q(s, a)]$ maximizes the Q-values of actions that actually exist in the dataset.
* **Standard Bellman Error:** The second term ensures the Q-function still satisfies the Bellman equation.


### Code Implementation
In this repository, the above math is directly translated into PyTorch operations. By simply toggling the `ALPHA_CQL` weight, the script smoothly transitions between standard SAC ($\alpha = 0$) and CQL ($\alpha > 0$).

```python
# 1. Standard Bellman Error (MSE)
mse_loss = F.mse_loss(q_pred, q_target)

# 2. CQL Penalty (Push Down OOD - Push Up Dataset)
# Sample 10 random actions per state to estimate the OOD space
rand_actions = torch.empty((BATCH_SIZE, 10, action_dim)).uniform_(-1, 1).to(device)
q_rand, _ = q_net(state_rep, rand_actions)

# LogSumExp (Push Down)
cql_logsumexp = torch.logsumexp(q_rand, dim=1).mean()
# Dataset actions (Push Up)
cql_dataset_q = q_pred.mean()

cql_loss = cql_logsumexp - cql_dataset_q

# 3. Final Objective
# If algo == 'sac', ALPHA_CQL is 0. If 'cql', ALPHA_CQL is 5.0.
total_q_loss = mse_loss + ALPHA_CQL * cql_loss
```



## 3. Experimental Setup
* **Environment:** `HalfCheetah-v4` (Gymnasium)
* **Dataset:** `halfcheetah-medium-v2` (Berkeley D4RL HDF5)
* **Hyperparameters:** Alpha=5.0 (for CQL), Learning Rate=3e-4, Batch Size=256, Epochs=100.



## 4. Results: WandB Learning Curves
The graphs below demonstrate the stability of CQL compared to the severe overestimation seen in SAC.
**Q-Value Estimates (Overestimation Bias)**
<img width="1088" height="640" alt="image" src="https://github.com/user-attachments/assets/663732f8-ad39-490c-aef1-8f372e21af02" />
**Evaluation Normalized Score (True Performance)**
<img width="1088" height="628" alt="image" src="https://github.com/user-attachments/assets/27c683a7-08c1-4ac9-bef3-b3d04a07921f" />
*(Note: SAC Q-values explode exponentially, leading to a complete collapse in actual evaluation performance. CQL maintains bounded Q-values and steadily improves evaluation scores.)*



## 5. Visual Simulation Comparison
To observe the physical manifestation of the learned policies, the agents were rendered in the MuJoCo physics engine. 


### Standard SAC (Failure due to OOD actions)
The SAC agent attempts actions with falsely optimistic Q-values, resulting in erratic, unstable behavior.
<img width="480" height="480" alt="sac_simulation" src="https://github.com/user-attachments/assets/84d5c77e-dde9-43de-bfc8-257330c2b369" />

### CQL (Stable conservative execution)
The CQL agent strictly adheres to behaviors close to the dataset distribution, successfully driving the HalfCheetah forward.
https://github.com/user-attachments/assets/a2524bd7-0891-457c-aaa0-5a74b8a88c76
<img width="480" height="480" alt="cql_simulation" src="https://github.com/user-attachments/assets/b1394d40-397a-4fa2-bf9d-f28844ab09b8" />



## 6. Reproduction
To reproduce these results, you can run the provided script with the respective algorithm flag:


```bash
# Install dependencies
pip install gymnasium[mujoco] wandb h5py

# Run standard SAC
python train.py --algo sac

# Run CQL
python train.py --algo cql

