"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) controller.
Contains Controller (inference), Agent (networks), and Trainer (training loop).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import random
from pathlib import Path

from .base_controller import BaseController
from utils.replay_memory import ReplayMemory
from utils.geometry import distance


# Neural Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Agent:
    def __init__(self, state_dim=3, action_dim=2, max_action=1.0, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.max_action = max_action
        self.training_mode = True
        
    def act(self, state, deterministic=False):
        """Get action from policy."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # NOTE: No exploration noise here. Exploration is handled in the Trainer
        # to avoid double-noise and to keep inference behavior clean.
        return np.clip(action, -self.max_action, self.max_action)
    
    def train_mode(self, mode=True):
        """Set training or evaluation mode."""
        self.training_mode = mode
        self.actor.train(mode)
        self.critic.train(mode)
    
    def save(self, filename):
        """Save model parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """Load model parameters."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


class TD3Controller(BaseController):
    def __init__(self, model_path=None):
        super().__init__()
        self.agent = None
        
        if model_path:
            self.load_model(model_path)
    
    def act(self, observation, dt=1.0/60.0):
        """
        Get action from TD3 policy (deterministic for inference).
        
        Args:
            observation: Current sensor readings as numpy array
            dt: Time delta for physics update (not used by TD3)
            
        Returns:
            [left_wheel_velocity, right_wheel_velocity] in range [-1, 1]
        """
        if self.agent is None:
            return [0.0, 0.0]
        
        # Ensure observation is numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Get deterministic action for inference
        action = self.agent.act(observation, deterministic=True)
        
        # Clamp to [-1, 1] range
        action = np.clip(action, -1.0, 1.0)
        
        return action.tolist()
    
    def load_model(self, model_path):
        """Load a trained TD3 model."""
        self.agent = TD3Agent()
        self.agent.load(model_path)
        self.agent.train_mode(False)  # Set to evaluation mode
        print(f"✓ Loaded TD3 model from {model_path}")
    
    def reset(self):
        """Reset controller state."""
        pass


class TD3Trainer:
    def __init__(self, env, config, save_dir, render=False):
        self.env = env
        self.config = config
        self.save_dir = Path(save_dir)
        self.render = render
        self._initial_render = render  # Remember if rendering was initially enabled
        
        # Training parameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.policy_delay = config['policy_delay']
        self.noise_sigma = config['noise_sigma']
        self.noise_clip = config['noise_clip']
        self.warmup_steps = config['warmup_steps']
        
        # Initialize agent and replay buffer
        self.agent = TD3Agent(lr=config['actor_lr'])
        self.replay_buffer = ReplayMemory(config['buffer_size'])
        
        # Training state
        self.total_steps = 0
        self.episode_num = 0
        
        # Best model tracking
        self.best_reward = float('-inf')  # Track highest reward achieved
        
        # Metrics tracking
        self.metrics = []

        # Periodic checkpointing
        self.save_every_episodes = int(config.get('save_every_episodes', 0) or 0)
        
    def train(self):
        """Main training loop."""
        print("Starting TD3 training...")
        
        max_episodes = self.config.get('episodes', 1000)
        max_steps = self.config.get('total_steps', None)
        
        while True:
            # Check termination conditions
            if max_episodes and self.episode_num >= max_episodes:
                break
            if max_steps and self.total_steps >= max_steps:
                break
            
            episode_reward, episode_steps, episode_distance = self._run_episode()
            
            self.episode_num += 1
            
            # Log episode metrics
            avg_speed = episode_distance / episode_steps if episode_steps > 0 else 0
            metrics = {
                'episode': self.episode_num,
                'total_reward': episode_reward,
                'steps': episode_steps,
                'total_distance': episode_distance,
                'avg_speed': avg_speed
            }
            self.metrics.append(metrics)
            
            # Print progress
            if self.episode_num % 10 == 0:
                print(f"Episode {self.episode_num}: Reward={episode_reward:.2f}, "
                      f"Steps={episode_steps}, Distance={episode_distance:.2f}, "
                      f"Avg Speed={avg_speed:.2f}")
            
            # Save best model if new best reward achieved
            self._save_best_model(episode_reward, self.episode_num)
            
            # Periodic checkpoint save
            if self.save_every_episodes > 0 and (self.episode_num % self.save_every_episodes == 0):
                self._save_checkpoint(self.episode_num)
        
        # Save final model and metrics
        self._save_final_model()
        self._save_metrics()
        
        print(f"Training completed! Total episodes: {self.episode_num}, Total steps: {self.total_steps}")
    
    def _run_episode(self):
        """Run a single episode and collect experience."""
        obs = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_distance = 0
        prev_pos = None
        epsilon = self.config.get('epsilon', 0.1)  # small probability of random action

        while True:
            # Get action
            if self.total_steps < self.warmup_steps:
                # Random actions during warmup
                action = np.random.uniform(-1, 1, 2)
            else:
                # Epsilon-greedy: sometimes take a random action to encourage turning
                if np.random.rand() < epsilon:
                    action = np.random.uniform(-1, 1, 2)
                else:
                    # Policy action (deterministic), then add Gaussian exploration noise
                    action = self.agent.act(obs, deterministic=True)
                    noise = np.random.normal(0, self.noise_sigma, size=action.shape)
                    noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                    action = np.clip(action + noise, -1, 1)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action, dt=1.0/60.0)
            
            # Calculate distance traveled
            current_pos = info['position']
            if prev_pos is not None:
                episode_distance += distance(prev_pos, current_pos)
            prev_pos = current_pos
            
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Store transition in replay buffer
            done = terminated or truncated
            self.replay_buffer.push(
                torch.FloatTensor(obs),
                torch.FloatTensor(action),
                torch.FloatTensor(next_obs),
                torch.FloatTensor([reward]),
                torch.FloatTensor([float(done)])
            )
            
            # Train agent if enough experience and past warmup
            if len(self.replay_buffer) > self.batch_size and self.total_steps > self.warmup_steps:
                self._update_agent()
            
            # Handle pygame events for toggling visibility (always process events if rendering was initially enabled)
            if hasattr(self, '_initial_render') and self._initial_render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            self.env.show_sensors = not self.env.show_sensors
                            print(f"Sensors: {'ON' if self.env.show_sensors else 'OFF'}")
                        elif event.key == pygame.K_c:
                            self.env.show_collision_box = not self.env.show_collision_box
                            print(f"Collision box: {'ON' if self.env.show_collision_box else 'OFF'}")
                        elif event.key == pygame.K_t:
                            self.env.show_track_edges = not self.env.show_track_edges
                            print(f"Track edges: {'ON' if self.env.show_track_edges else 'OFF'}")
                        elif event.key == pygame.K_r:
                            self.env.show_racing_line = not self.env.show_racing_line
                            print(f"Racing line: {'ON' if self.env.show_racing_line else 'OFF'}")
                        elif event.key == pygame.K_h:
                            self.render = not self.render
                            print(f"Rendering: {'ON' if self.render else 'OFF'}")
                
                # Only render if currently enabled
                if self.render:
                    self.env.render()
                    pygame.display.flip()
                    # Control frame rate during rendering
                    pygame.time.wait(16)  # ~60 FPS
            
            obs = next_obs
            
            if done:
                break
        
        return episode_reward, episode_steps, episode_distance
    
    def _update_agent(self):
        """Update TD3 agent with a batch of experience."""
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        state_batch = state_batch.to(self.agent.device)
        action_batch = action_batch.to(self.agent.device)
        reward_batch = reward_batch.to(self.agent.device)
        next_state_batch = next_state_batch.to(self.agent.device)
        done_batch = done_batch.to(self.agent.device)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action_batch) * self.noise_sigma).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.agent.actor_target(next_state_batch) + noise).clamp(-self.agent.max_action, self.agent.max_action)
            
            # Compute target Q values
            target_Q1, target_Q2 = self.agent.critic_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.agent.critic(state_batch, action_batch)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize critic
        self.agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.agent.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_steps % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.agent.critic.Q1(state_batch, self.agent.actor(state_batch)).mean()
            
            # Optimize actor
            self.agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.agent.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.agent.critic.parameters(), self.agent.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.agent.actor.parameters(), self.agent.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _save_final_model(self):
        """Save the final trained model."""
        model_path = self.save_dir / "final.pt"
        self.agent.save(model_path)
        print(f"✓ Final model saved to {model_path}")
    
    def _save_checkpoint(self, episode_num: int):
        """Save a periodic checkpoint of the model."""
        ckpt_path = self.save_dir / f"checkpoint_ep{episode_num}.pt"
        self.agent.save(ckpt_path)
        print(f"✓ Checkpoint saved at episode {episode_num} to {ckpt_path}")
    
    def _save_best_model(self, episode_reward: float, episode_num: int):
        """Save model if it achieves a new best reward."""
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            best_path = self.save_dir / "best_model.pt"
            self.agent.save(best_path)
            print(f"✓ New best model saved at episode {episode_num} with reward {episode_reward:.2f} to {best_path}")
            return True
        return False
    
    def _save_metrics(self):
        """Save training metrics to CSV."""
        metrics_path = self.save_dir / "metrics.csv"
        
        with open(metrics_path, 'w', newline='') as csvfile:
            fieldnames = ['episode', 'total_reward', 'steps', 'total_distance', 'avg_speed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metrics in self.metrics:
                writer.writerow(metrics)
        
        print(f"✓ Metrics saved to {metrics_path}")
