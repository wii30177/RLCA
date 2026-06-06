#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO for Hopper-v5 — Factorized-RLCA full joint config (11³ = 1331 logits)

基於 ppo_hopper-v5.py（連續 Gaussian baseline）的最小修改版，只動四處：
  (1) DiscretizeHopperAction wrapper（Discrete(1331) + LUT，11 進位 3 位數拆解）
  (2) Actor 輸出層：Gaussian mean/log_std → 單顆 1331-logits Categorical head
  (3) test()：取 mean → argmax
  (4) wandb / checkpoint 命名
其餘訓練迴圈、GAE、超參數與 baseline 完全相同（對照實驗只差動作頭結構）。
Architecture follows ddpg.py conventions.

Key differences from DDPG:
  - On-policy: rollout data is discarded after each update (no replay buffer)
  - Clipped surrogate loss replaces deterministic policy gradient
  - GAE (Generalized Advantage Estimation) replaces one-step TD
  - No target networks needed (no off-policy bootstrapping issue)
  - Entropy bonus for exploration (no additive OU noise)
  - Single optimizer for actor + critic
"""

import gymnasium as gym
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb
from tqdm import tqdm

# ─── Action discretization wrapper (Factorized-RLCA) ──────────────────────────
class DiscretizeHopperAction(gym.ActionWrapper):
    """
    Hopper-v5 的 3-D 連續動作 → Discrete(bins**3)（full joint, 1331 logits）。
    對外動作為 0..1330 的整數，視為 11 進位 3 位數：(thigh, leg, foot)。
    每個 joint 離散到 bins 個值（預設 [-1.0, -0.8, ..., 1.0]），索引經查表 (LUT) 轉成實際 torque。
    """
    def __init__(self, env, bins=11, low=-1.0, high=1.0):
        super().__init__(env)
        self.bins = bins
        # 11 個離散值 [-1.0, -0.8, ..., 1.0]，索引→實際 torque 的查表
        self.table = np.linspace(low, high, bins, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(bins ** 3)

    def action(self, act):
        # act: 純量或 shape (1,) 的整數，11 進位 3 位數 → (thigh, leg, foot)
        index     = int(np.asarray(act).reshape(-1)[0])
        thigh_idx = index // (self.bins ** 2)        # 高位 → thigh
        leg_idx   = (index // self.bins) % self.bins  # 中位 → leg
        foot_idx  = index % self.bins                 # 低位 → foot
        return np.array(
            [self.table[thigh_idx], self.table[leg_idx], self.table[foot_idx]],
            dtype=np.float32,
        )


# ─── Utility: seed wrapper (identical to ddpg.py) ─────────────────────────────
class SeedWrapper(gym.Wrapper):
    """Applies a fixed seed on the first reset; subsequent resets are random."""
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed
        self._used = False  # only apply fixed seed on the very first reset

    def reset(self, seed=None):
        if seed is not None:
            return self.env.reset(seed=seed)
        if not self._used:
            self._used = True
            return self.env.reset(seed=self._seed)
        return self.env.reset()  # subsequent resets: random starts


# ─── Rollout Buffer (on-policy, replaces DDPG's ReplayMemory) ─────────────────
class RolloutBuffer:
    """
    Collects transitions for one PPO rollout (n_steps).

    Unlike DDPG's ReplayMemory (off-policy, large, randomly sampled),
    this buffer is always cleared after every policy update because
    PPO can only reuse data collected under the *current* policy.
    """
    def __init__(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.dones     = []   # True only on real termination (not timeout truncation)
        self.log_probs = []
        self.values    = []

    def push(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        """Discard all collected data (called after each PPO update)."""
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.dones     = []
        self.log_probs = []
        self.values    = []

    def __len__(self):
        return len(self.states)


# ─── Actor: flat categorical policy over the full joint action space ──────────
class Actor(nn.Module):
    """
    Full joint categorical policy: 單顆 head 輸出 11³ = 1331 logits，
    一次選定 (thigh, leg, foot) 的完整組合（最大的搜尋空間，對照組）。

    為了與 factorized 版本共用相同的程式結構，仍以「head 列表」的形式實作
    （此處列表長度為 1），action 為 shape [..., 1] 的索引張量。

    At test time, take the argmax of the head (deterministic greedy policy).
    """
    def __init__(self, hidden_size, num_inputs, action_space):
        super().__init__()
        nvec = [int(action_space.n)]  # [1331]

        # Two hidden layers (same size as ddpg.py)
        self.fc1   = nn.Linear(num_inputs, hidden_size)
        self.fc2   = nn.Linear(hidden_size, hidden_size)
        self.heads = nn.ModuleList(nn.Linear(hidden_size, n) for n in nvec)

        # Orthogonal initialization: recommended for PPO (better gradient flow
        # than Kaiming in early training for tanh activations)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=0.01)  # small final-layer gain

    def forward(self, x):
        """Return a list of logits tensors, one per action head."""
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return [head(x) for head in self.heads]

    def get_action_and_log_prob(self, x):
        """
        Sample an action index from each head; log_prob is summed across heads.
        Used during rollout collection (exploration).
        """
        logits   = self.forward(x)
        dists    = [Categorical(logits=l) for l in logits]
        actions  = [d.sample() for d in dists]
        log_prob = sum(d.log_prob(a) for d, a in zip(dists, actions))
        action   = torch.stack(actions, dim=-1)  # shape: [..., n_heads]
        return action, log_prob

    def evaluate_actions(self, x, actions):
        """
        Compute log π_θ(a|s) and entropy for given (state, action-index) pairs.
        Used during the PPO update to re-evaluate old actions under new θ.
        """
        logits   = self.forward(x)
        dists    = [Categorical(logits=l) for l in logits]
        log_prob = sum(d.log_prob(actions[..., i]) for i, d in enumerate(dists))
        entropy  = sum(d.entropy() for d in dists)
        return log_prob, entropy


# ─── Critic: state-value function V(s) ────────────────────────────────────────
class Critic(nn.Module):
    """
    Value network V(s) — estimates expected cumulative discounted reward from state s.

    Unlike DDPG's Q(s,a) critic, PPO's critic takes only the state as input.
    It is used to compute advantages (GAE) and as a regression target during update.
    """
    def __init__(self, hidden_size, num_inputs):
        super().__init__()
        self.fc1  = nn.Linear(num_inputs, hidden_size)
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

        nn.init.orthogonal_(self.fc1.weight,  gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight,  gain=np.sqrt(2))
        nn.init.orthogonal_(self.head.weight, gain=1.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.head(x)  # shape: [batch, 1]


# ─── PPO Agent ─────────────────────────────────────────────────────────────────
class PPO:
    """
    Proximal Policy Optimization agent.

    The core idea: after collecting a rollout with the current policy π_old,
    run K epochs of gradient updates on a *clipped* objective that prevents
    the updated policy π_θ from moving too far from π_old in a single step.

    Clipped objective:
        L_CLIP = E[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]
    where r_t = π_θ(a|s) / π_old(a|s)  (importance sampling ratio)

    This is combined with a value-function loss and an optional entropy bonus.
    """
    def __init__(self, num_inputs, action_space, gamma=0.99, gae_lambda=0.95,
                 hidden_size=256, lr=3e-4, clip_epsilon=0.2,
                 n_epochs=10, value_coef=0.5, entropy_coef=0.0,
                 max_grad_norm=0.5, mini_batch_size=64):

        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_epsilon    = clip_epsilon
        self.n_epochs        = n_epochs
        self.value_coef      = value_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm
        self.mini_batch_size = mini_batch_size

        self.actor  = Actor(hidden_size, num_inputs, action_space)
        self.critic = Critic(hidden_size, num_inputs)

        # Single optimizer for actor + critic (common PPO practice;
        # allows joint learning-rate scheduling if desired)
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def select_action(self, state):
        """
        Sample action stochastically during rollout collection.
        Returns action, its log probability, and current state value.
        """
        self.actor.eval()
        self.critic.eval()

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(state)
            value            = self.critic(state)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        Generalized Advantage Estimation (GAE-Lambda).

        GAE smoothly interpolates between:
          - lambda=0: TD(0) advantage (low variance, high bias)
          - lambda=1: Monte Carlo advantage (high variance, low bias)

        At each step t (going backwards):
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t     = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}

        The (1 - done_t) term zeroes the bootstrap at episode boundaries,
        preventing value estimates from bleeding across episodes.
        """
        advantages      = []
        gae             = 0.0
        extended_values = values + [last_value]  # append bootstrap value at end

        for t in reversed(range(len(rewards))):
            # One-step TD error (delta)
            delta = (rewards[t]
                     + self.gamma * extended_values[t + 1] * (1.0 - dones[t])
                     - extended_values[t])
            # Accumulate GAE backwards
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        # Returns = V(s) + A(s,a) ≈ expected cumulative reward from (s,a)
        returns    = advantages + torch.FloatTensor(values)
        return advantages, returns

    def update_parameters(self, buffer, last_value):
        """
        Perform PPO update on collected rollout data.

        Steps:
          1. Compute GAE advantages and target returns.
          2. Normalize advantages (zero mean, unit std) for training stability.
          3. Repeat n_epochs times: shuffle data, iterate over mini-batches,
             compute clipped loss, back-propagate with gradient clipping.
        """
        self.actor.train()
        self.critic.train()

        # Convert buffer data to tensors
        states        = torch.FloatTensor(np.array(buffer.states))
        actions       = torch.stack(buffer.actions)
        old_log_probs = torch.stack(buffer.log_probs).detach()  # detach: these are fixed targets
        raw_values    = [v.item() for v in buffer.values]

        advantages, returns = self._compute_gae(
            buffer.rewards, raw_values, buffer.dones, last_value
        )
        # Normalize advantages: reduces sensitivity to reward scale
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples         = len(buffer)
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        num_mini_updates  = 0

        for _ in range(self.n_epochs):
            # Shuffle mini-batch order each epoch to reduce gradient correlation
            perm = torch.randperm(n_samples)

            for start in range(0, n_samples, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]

                mb_states    = states[idx]
                mb_actions   = actions[idx]
                mb_old_lp    = old_log_probs[idx]
                mb_adv       = advantages[idx]
                mb_returns   = returns[idx]

                # Re-evaluate actions under the *current* (updated) policy θ
                new_log_probs, entropy = self.actor.evaluate_actions(mb_states, mb_actions)
                values_pred = self.critic(mb_states).squeeze(-1)

                # Importance sampling ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_log_probs - mb_old_lp)

                # ── PPO clipped surrogate loss ─────────────────────────────
                # Without clipping: policy can take arbitrarily large steps.
                # Clipping to [1-ε, 1+ε] creates a trust region around π_old.
                surr1       = ratio * mb_adv
                surr2       = torch.clamp(ratio,
                                          1 - self.clip_epsilon,
                                          1 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()  # maximize clipped objective

                # ── Value function loss (MSE) ──────────────────────────────
                # Train V(s) to predict bootstrapped returns
                value_loss = F.mse_loss(values_pred, mb_returns)

                # ── Entropy bonus ──────────────────────────────────────────
                # Maximize entropy to encourage exploration;
                # negative sign because we minimize loss
                entropy_loss = -entropy.mean()

                # Combined loss (same convention as ddpg.py combined update)
                loss = (policy_loss
                        + self.value_coef  * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents exploding gradients (important for PPO)
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                num_mini_updates  += 1

        n = max(num_mini_updates, 1)
        return total_policy_loss / n, total_value_loss / n, total_entropy / n

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        os.makedirs('preTrained', exist_ok=True)
        if actor_path is None:
            actor_path  = f"preTrained/ppo_actor_logits1331_{env_name}_{suffix}"
        if critic_path is None:
            critic_path = f"preTrained/ppo_critic_logits1331_{env_name}_{suffix}"
        print(f'Saving models to {actor_path} and {critic_path}')
        torch.save(self.actor.state_dict(),  actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        return actor_path, critic_path

    def load_model(self, actor_path, critic_path):
        print(f'Loading models from {actor_path} and {critic_path}')
        if actor_path  is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


# ─── Training ──────────────────────────────────────────────────────────────────
def train():
    # ── Hyperparameters ──────────────────────────────────────────────────────
    total_steps     = 1_000_000   # total env steps (≈488 updates at n_steps=2048)
    gamma           = 0.99        # discount factor
    gae_lambda      = 0.95        # GAE-Lambda (balance bias vs. variance)
    hidden_size     = 256         # neurons per hidden layer (same as ddpg.py)
    lr              = 3e-4        # learning rate shared by actor and critic
    clip_epsilon    = 0.2         # PPO clip range (standard value from paper)
    n_epochs        = 10          # gradient epochs per rollout (K in the paper)
    mini_batch_size = 64          # mini-batch size within each epoch
    value_coef      = 0.5         # weight on value loss (c1 in the paper)
    entropy_coef    = 0.0         # entropy bonus weight (Hopper doesn't need much)
    max_grad_norm   = 0.5         # gradient norm clipping
    n_steps         = 2048        # rollout length before each PPO update
    print_freq      = 10          # log/print every N completed episodes
    eval_freq       = 50_000      # evaluate and save model every N env steps
    # ─────────────────────────────────────────────────────────────────────────

    ewma_reward     = 0.0
    rewards         = []
    total_numsteps  = 0
    updates         = 0
    episode_count   = 0
    last_eval_step  = 0           # tracks when the last eval/save was triggered
    p_loss = v_loss = ent = 0.0   # last known losses (for logging before first update)

    # ── W&B: initialize run ───────────────────────────────────────────────────
    wandb.init(
        project="ppo",
        name=f"{env_name}_logits1331_seed{random_seed}",
        config={
            "env":             env_name,
            "variant":         "logits1331_full",
            "bins":            bins,
            "n_logits":        bins ** 3,   # 11³ = 1331
            "seed":            random_seed,
            "total_steps":     total_steps,
            "gamma":           gamma,
            "gae_lambda":      gae_lambda,
            "hidden_size":     hidden_size,
            "lr":              lr,
            "clip_epsilon":    clip_epsilon,
            "n_epochs":        n_epochs,
            "mini_batch_size": mini_batch_size,
            "value_coef":      value_coef,
            "entropy_coef":    entropy_coef,
            "max_grad_norm":   max_grad_norm,
            "n_steps":         n_steps,
            "eval_freq":       eval_freq,
        },
    )
    # ─────────────────────────────────────────────────────────────────────────

    agent = PPO(
        env.observation_space.shape[0], env.action_space,
        gamma=gamma, gae_lambda=gae_lambda, hidden_size=hidden_size,
        lr=lr, clip_epsilon=clip_epsilon, n_epochs=n_epochs,
        value_coef=value_coef, entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm, mini_batch_size=mini_batch_size
    )
    buffer = RolloutBuffer()

    # ── W&B: watch networks for gradient/weight tracking ─────────────────────
    #wandb.watch(agent.actor,  log="all", log_freq=100, idx=0)
    #wandb.watch(agent.critic, log="all", log_freq=100, idx=1)
    # ─────────────────────────────────────────────────────────────────────────

    # Initialize environment state (PPO collects continuously across episodes)
    state, _       = env.reset()
    episode_reward = 0.0

    pbar = tqdm(total=total_steps, desc="PPO Training")

    while total_numsteps < total_steps:

        # ── Collect n_steps transitions (may cross episode boundaries) ────────
        # Unlike DDPG (per-episode loop), PPO collects a fixed number of steps
        # regardless of episode boundaries, then performs a batch update.
        for _ in range(n_steps):
            state_t = torch.FloatTensor(np.array(state))
            action, log_prob, value = agent.select_action(state_t)

            next_state, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated

            # Store 'terminated' (not 'done') as the done flag for GAE:
            # truncation is a timeout, not a true terminal — we should still bootstrap
            buffer.push(state_t.numpy(), action, reward, float(terminated),
                        log_prob, value)

            state          = next_state
            episode_reward += reward
            total_numsteps += 1
            pbar.update(1)

            if done:
                # Episode finished: record and reset environment
                rewards.append(episode_reward)
                episode_count += 1

                if episode_count % print_freq == 0:
                    print(f"Episode {episode_count} | steps {total_numsteps} "
                          f"| reward {episode_reward:.2f}")
                    wandb.log({
                        "episode":           episode_count,
                        "train/total_steps": total_numsteps,
                        "train/updates":     updates,
                        "train/reward":      episode_reward,
                        "train/policy_loss": p_loss,
                        "train/value_loss":  v_loss,
                        "train/entropy":     ent,
                    }, step=episode_count)

                # Evaluate and save every eval_freq steps (step-based, not episode-based)
                if total_numsteps - last_eval_step >= eval_freq:
                    last_eval_step = total_numsteps
                    suffix = f"step{total_numsteps}_ep{episode_count}.pth"
                    actor_path, critic_path = agent.save_model(env_name, suffix)
                    mean_eval = test(actor_path, critic_path)

                    wandb.log({
                        "train/total_steps": total_numsteps,
                        "eval/mean_reward":  mean_eval,
                    }, step=episode_count)

                    artifact = wandb.Artifact(
                        name=f"ppo-{env_name}-checkpoint",
                        type="model",
                        description=f"step={total_numsteps}, episode={episode_count}",
                    )
                    artifact.add_dir("preTrained/")
                    wandb.log_artifact(artifact)

                episode_reward = 0.0
                state, _       = env.reset()

            if total_numsteps >= total_steps:
                break

        # ── PPO Update ────────────────────────────────────────────────────────
        # Bootstrap value for the last collected state:
        # If episode ended on a true terminal, use 0; otherwise V(s_T).
        with torch.no_grad():
            last_val = agent.critic(
                torch.FloatTensor(np.array(state)).unsqueeze(0)
            ).item()

        p_loss, v_loss, ent = agent.update_parameters(buffer, last_val)
        buffer.clear()   # discard rollout (on-policy: cannot reuse old data)
        updates += 1

    pbar.close()
    wandb.finish()


# ─── Test ──────────────────────────────────────────────────────────────────────
def test(actor_path, critic_path, hidden_size=256, n_episodes=20, render=False):
    """Test the learned model (no change needed)."""
    test_env = DiscretizeHopperAction(
        gym.make(env_name, render_mode="human" if render else None), bins=bins
    )
    model    = PPO(test_env.observation_space.shape[0], test_env.action_space,
                   hidden_size=hidden_size)
    model.load_model(actor_path, critic_path)

    eval_reward_history = []

    for i_episode in range(1, n_episodes + 1):
        state, _ = test_env.reset()
        running_reward = 0
        t = 0

        while True:
            # Deterministic evaluation: argmax of each categorical head
            with torch.no_grad():
                logits = model.actor(
                    torch.FloatTensor(np.array(state)).unsqueeze(0)
                )
            action = np.array([l.argmax(dim=-1).item() for l in logits],
                              dtype=np.int64)

            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            running_reward += reward
            t += 1
            if render:
                test_env.render() 
            if done:
                eval_reward_history.append(running_reward)
                print(f"Eval Episode: {i_episode}, length: {t}, reward: {running_reward:.2f}")
                break

    mean_reward = np.mean(eval_reward_history)
    print(f'Number of Eval Episodes: {n_episodes}\t; Evaluation Reward: {mean_reward}')
    test_env.close()
    return mean_reward


# ─── Reproducibility (same as ddpg.py) ────────────────────────────────────────
def set_seed(env, seed):
    """Fix random seed across all libraries for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True
    env = SeedWrapper(env=env, seed=seed)
    env.action_space.seed(seed)


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    random_seed = 42
    env_name    = 'Hopper-v5'
    bins        = 11
    env = DiscretizeHopperAction(gym.make(env_name), bins=bins)
    set_seed(env, seed=random_seed)
    train()
    # To run evaluation only:
    # test("preTrained/ppo_actor_logits1331_Hopper-v5_stepXXX_epXXX.pth",
    #      "preTrained/ppo_critic_logits1331_Hopper-v5_stepXXX_epXXX.pth",
    #      render=True)
