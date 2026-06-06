#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAC (Soft Actor-Critic) for Hopper-v5
Architecture follows ppo_hopper-v5.py conventions.

Key differences from PPO:
  - Off-policy: replay buffer retains data for reuse (no rollout discard)
  - Two Q-critics (clipped double-Q) to reduce value overestimation
  - Target networks for Q-critics with soft Polyak updates (no hard copy)
  - Entropy-regularized objective: maximize E[r] + alpha * H[pi]
  - Automatic temperature (alpha) tuning via dual gradient descent
  - Squashed Gaussian policy (tanh reparameterization) for bounded actions
  - State-dependent log_std per action dim (richer than PPO's shared param)
  - Separate optimizers for actor, critics, and alpha
"""

import gymnasium as gym
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal
import wandb
from tqdm import tqdm

LOG_STD_MAX =  2
LOG_STD_MIN = -20


# ─── Utility: seed wrapper (identical to ppo_hopper-v5.py) ────────────────────
class SeedWrapper(gym.Wrapper):
    """Applies a fixed seed on the first reset; subsequent resets are random."""
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed
        self._used = False

    def reset(self, seed=None):
        if seed is not None:
            return self.env.reset(seed=seed)
        if not self._used:
            self._used = True
            return self.env.reset(seed=self._seed)
        return self.env.reset()


# ─── Replay Buffer (off-policy, replaces PPO's RolloutBuffer) ─────────────────
class ReplayBuffer:
    """
    Circular replay buffer for off-policy SAC.

    Unlike PPO's RolloutBuffer (cleared after every update because PPO is
    on-policy), this buffer retains up to `capacity` transitions and supports
    random mini-batch sampling so the same experience can be reused many times.
    Pre-allocated numpy arrays avoid Python-list overhead of PPO's buffer.
    """
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.pos      = 0
        self.size     = 0

        self.states      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos]      = state
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos]       = float(done)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]),
        )

    def __len__(self):
        return self.size


# ─── Actor: Squashed Gaussian policy ──────────────────────────────────────────
class Actor(nn.Module):
    """
    Stochastic policy that outputs a squashed Gaussian distribution.

    Unlike PPO's actor (shared state-independent log_std), SAC uses both a
    state-dependent mean AND log_std so the distribution adapts to each state.
    Actions are sampled via the reparameterization trick and squashed with tanh
    to respect the environment's action bounds.

    Squashed Gaussian: a = tanh(u),  u ~ N(mean(s), std(s))
    Log-prob correction (Appendix C, Haarnoja et al. 2018):
        log pi(a|s) = log N(u; mean, std) - sum_i log(1 - tanh^2(u_i))
    """
    def __init__(self, hidden_size, num_inputs, action_space):
        super().__init__()
        num_outputs = action_space.shape[0]

        # Buffers for rescaling tanh output from [-1,1] to actual action range
        action_high = torch.FloatTensor(action_space.high)
        action_low  = torch.FloatTensor(action_space.low)
        self.register_buffer('action_scale', (action_high - action_low) / 2.0)
        self.register_buffer('action_bias',  (action_high + action_low) / 2.0)

        self.fc1          = nn.Linear(num_inputs, hidden_size)
        self.fc2          = nn.Linear(hidden_size, hidden_size)
        self.mean_head    = nn.Linear(hidden_size, num_outputs)
        self.log_std_head = nn.Linear(hidden_size, num_outputs)  # state-dependent

    def forward(self, x):
        """Return mean and clamped log_std of the pre-squash Gaussian."""
        x       = F.relu(self.fc1(x))
        x       = F.relu(self.fc2(x))
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        """
        Sample action via reparameterization and compute its log probability.

        Reparameterization (u = mean + eps * std, eps ~ N(0,I)) makes the
        sample differentiable w.r.t. actor parameters — required for the
        actor gradient update (unlike PPO which uses REINFORCE).
        """
        mean, log_std = self.forward(x)
        std  = log_std.exp()
        dist = Normal(mean, std)

        u      = dist.rsample()                         # reparameterized sample
        action = torch.tanh(u)

        # Tanh squashing changes the distribution; correct the log-prob:
        #   log pi(a|s) = log N(u|s) - sum(log(1 - tanh^2(u)))
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Scale action from [-1, 1] to actual environment action range
        action = action * self.action_scale + self.action_bias
        return action, log_prob

    def get_action_deterministic(self, x):
        """Deterministic action for evaluation: mean of distribution, squashed."""
        mean, _ = self.forward(x)
        action  = torch.tanh(mean)
        return action * self.action_scale + self.action_bias


# ─── Critic: Double Q-function Q(s, a) ────────────────────────────────────────
class Critic(nn.Module):
    """
    Two independent Q(s, a) networks implementing clipped double-Q.

    Unlike PPO's V(s) critic (state only), SAC's critic takes both state and
    action as input (same as DDPG). Having two independent critics and using
    the *minimum* during target computation prevents the systematic Q-value
    overestimation that arises from maximizing a single noisy Q estimate.
    """
    def __init__(self, hidden_size, num_inputs, num_actions):
        super().__init__()
        in_dim = num_inputs + num_actions

        # Q1
        self.q1_fc1  = nn.Linear(in_dim, hidden_size)
        self.q1_fc2  = nn.Linear(hidden_size, hidden_size)
        self.q1_head = nn.Linear(hidden_size, 1)

        # Q2 (independent weights to decorrelate the two estimates)
        self.q2_fc1  = nn.Linear(in_dim, hidden_size)
        self.q2_fc2  = nn.Linear(hidden_size, hidden_size)
        self.q2_head = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x  = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_head(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_head(q2)

        return q1, q2


# ─── SAC Agent ─────────────────────────────────────────────────────────────────
class SAC:
    """
    Soft Actor-Critic agent.

    SAC maximizes an entropy-regularized objective:
        J(pi) = E[ sum_t  r_t + alpha * H[pi(·|s_t)] ]

    The temperature alpha controls the exploration-exploitation trade-off:
    high alpha → explore (high entropy), low alpha → exploit (near-greedy).
    With auto-tuning, alpha adjusts automatically to maintain a target entropy
    level (heuristic: -dim(action_space), Haarnoja et al. 2018).

    Per-step update order:
      1. Critic update: Bellman error with entropy-regularized targets
      2. Actor update: maximize min(Q1, Q2) - alpha * log_pi
      3. Alpha update: dual gradient descent on the entropy constraint
      4. Soft update target critics via Polyak averaging
    """
    def __init__(self, num_inputs, action_space, gamma=0.99, tau=0.005,
                 alpha=0.2, hidden_size=256, lr=3e-4, auto_alpha=True,
                 target_entropy=None):

        self.gamma      = gamma
        self.tau        = tau           # Polyak soft-update coefficient
        self.auto_alpha = auto_alpha

        self.actor         = Actor(hidden_size, num_inputs, action_space)
        self.critic        = Critic(hidden_size, num_inputs, action_space.shape[0])
        # Target critics: never trained by gradient; soft-updated from critic
        self.critic_target = Critic(hidden_size, num_inputs, action_space.shape[0])
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False     # target params updated only via Polyak

        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        # Auto-tuning alpha: treat log_alpha as a learnable parameter and
        # optimize the soft entropy constraint via dual gradient descent.
        if auto_alpha:
            # Target entropy heuristic: -dim(A) (Haarnoja et al. 2018, App. D)
            self.target_entropy  = (target_entropy if target_entropy is not None
                                    else -float(action_space.shape[0]))
            self.log_alpha       = torch.zeros(1, requires_grad=True)
            self.alpha           = self.log_alpha.exp().item()
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

    def select_action(self, state, evaluate=False):
        """
        Stochastic action during rollout collection; deterministic for evaluation.
        Unlike PPO (which always samples), SAC evaluation uses the policy mean.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                action = self.actor.get_action_deterministic(state)
            else:
                action, _ = self.actor.sample(state)

        return action.squeeze(0).numpy()

    def update_parameters(self, buffer, batch_size):
        """
        One full SAC update from a random mini-batch.

        Returns scalar losses for W&B logging.
        """
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # ── Critic update ──────────────────────────────────────────────────────
        # Entropy-regularized Bellman backup (no gradient through targets):
        #   y = r + gamma * (1 - done) * (min_Q(s', a') - alpha * log_pi(a'|s'))
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            q1_next, q2_next          = self.critic_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            q_target   = rewards + self.gamma * (1 - dones) * (min_q_next - self.alpha * next_log_pi)

        q1, q2      = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor update ───────────────────────────────────────────────────────
        # Freeze critic to skip computing its useless gradients during actor step
        for p in self.critic.parameters():
            p.requires_grad = False

        pi, log_pi   = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, pi)
        min_q_pi     = torch.min(q1_pi, q2_pi)
        # Maximize E[Q(s, a) - alpha * log pi(a|s)]  →  minimize the negative
        actor_loss   = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # ── Alpha (temperature) update ─────────────────────────────────────────
        # Dual gradient descent: drive H[pi] toward target_entropy.
        # Loss: -log_alpha * (log_pi + target_entropy)  [stop-gradient on log_pi]
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ── Soft update target critics (Polyak) ───────────────────────────────
        # theta_target ← tau * theta + (1 - tau) * theta_target
        with torch.no_grad():
            for p, p_tgt in zip(self.critic.parameters(),
                                 self.critic_target.parameters()):
                p_tgt.data.mul_(1 - self.tau)
                p_tgt.data.add_(self.tau * p.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), self.alpha

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        os.makedirs('preTrained', exist_ok=True)
        if actor_path is None:
            actor_path  = f"preTrained/sac_actor_{env_name}_{suffix}"
        if critic_path is None:
            critic_path = f"preTrained/sac_critic_{env_name}_{suffix}"
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
    total_steps      = 1_000_000    # total env steps
    gamma            = 0.99         # discount factor
    tau              = 0.005        # Polyak soft-update coefficient
    alpha            = 0.2          # initial temperature (ignored if auto_alpha)
    auto_alpha       = True         # automatically tune alpha
    hidden_size      = 256          # neurons per hidden layer (same as ppo_hopper-v5.py)
    lr               = 3e-4         # learning rate for actor, critic, and alpha
    batch_size       = 256          # mini-batch size per gradient update
    replay_size      = 1_000_000    # replay buffer capacity
    start_steps      = 10_000       # random exploration steps before learning
    updates_per_step = 1            # gradient updates per env step
    print_freq       = 10           # log every N completed episodes
    eval_freq        = 50_000       # evaluate and save every N env steps
    # ─────────────────────────────────────────────────────────────────────────

    rewards        = []
    total_numsteps = 0
    updates        = 0
    episode_count  = 0
    last_eval_step = 0
    c_loss = a_loss = al_loss = 0.0
    alpha_val = alpha

    # ── W&B: initialize run ───────────────────────────────────────────────────
    wandb.init(
        project="sac",
        name=f"{env_name}_seed{random_seed}",
        config={
            "env":              env_name,
            "seed":             random_seed,
            "total_steps":      total_steps,
            "gamma":            gamma,
            "tau":              tau,
            "alpha":            alpha,
            "auto_alpha":       auto_alpha,
            "hidden_size":      hidden_size,
            "lr":               lr,
            "batch_size":       batch_size,
            "replay_size":      replay_size,
            "start_steps":      start_steps,
            "updates_per_step": updates_per_step,
            "eval_freq":        eval_freq,
        },
    )
    # ─────────────────────────────────────────────────────────────────────────

    num_inputs  = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    agent  = SAC(num_inputs, env.action_space,
                 gamma=gamma, tau=tau, alpha=alpha,
                 hidden_size=hidden_size, lr=lr, auto_alpha=auto_alpha)
    buffer = ReplayBuffer(replay_size, num_inputs, num_actions)

    wandb.watch(agent.actor,  log="all", log_freq=100, idx=0)
    wandb.watch(agent.critic, log="all", log_freq=100, idx=1)

    pbar = tqdm(total=total_steps, desc="SAC Training")

    for i_episode in range(1, 100_000):
        episode_reward = 0.0
        state, _       = env.reset()

        while True:
            # Random uniform actions during warm-up to fill the buffer
            # with diverse experience before gradient updates begin
            if total_numsteps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store 'terminated' (not 'done') as the done flag:
            # truncation is a timeout, not a true terminal — we should still
            # bootstrap the value, same reasoning as PPO's buffer.push()
            buffer.push(state, action, reward, next_state, float(terminated))

            state          = next_state
            episode_reward += reward
            total_numsteps += 1
            pbar.update(1)

            # ── Update: one gradient step per env step (off-policy) ───────────
            if len(buffer) >= batch_size and total_numsteps >= start_steps:
                for _ in range(updates_per_step):
                    c_loss, a_loss, al_loss, alpha_val = agent.update_parameters(
                        buffer, batch_size
                    )
                    updates += 1

            # ── Evaluate and checkpoint ────────────────────────────────────────
            if total_numsteps - last_eval_step >= eval_freq:
                last_eval_step = total_numsteps
                suffix         = f"step{total_numsteps}_ep{episode_count}.pth"
                actor_path, critic_path = agent.save_model(env_name, suffix)
                mean_eval = test(actor_path, critic_path)

                wandb.log({
                    "train/total_steps": total_numsteps,
                    "eval/mean_reward":  mean_eval,
                }, step=episode_count)

                artifact = wandb.Artifact(
                    name=f"sac-{env_name}-checkpoint",
                    type="model",
                    description=f"step={total_numsteps}, episode={episode_count}",
                )
                artifact.add_dir("preTrained/")
                wandb.log_artifact(artifact)

            if done or total_numsteps >= total_steps:
                break

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
                "train/critic_loss": c_loss,
                "train/actor_loss":  a_loss,
                "train/alpha_loss":  al_loss,
                "train/alpha":       alpha_val,
            }, step=episode_count)

        if total_numsteps >= total_steps:
            break

    pbar.close()
    wandb.finish()


# ─── Test ──────────────────────────────────────────────────────────────────────
def test(actor_path, critic_path, hidden_size=256, n_episodes=20, render=False):
    """Test the learned policy (deterministic mean action, no exploration)."""
    test_env = gym.make(env_name, render_mode="human" if render else None)
    model    = SAC(test_env.observation_space.shape[0], test_env.action_space,
                   hidden_size=hidden_size)
    model.load_model(actor_path, critic_path)

    eval_reward_history = []

    for i_episode in range(1, n_episodes + 1):
        state, _       = test_env.reset()
        running_reward = 0
        t              = 0

        while True:
            action = model.select_action(state, evaluate=True)
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


# ─── Reproducibility (same as ppo_hopper-v5.py) ───────────────────────────────
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
    env = gym.make(env_name, render_mode=None)
    set_seed(env, seed=random_seed)
    # train()
    # To run evaluation only:
    test("preTrained/sac_actor_Hopper-v5_step600000_ep1716.pth",
         "preTrained/sac_critic_Hopper-v5_step600000_ep1716.pth",
         render=True)
