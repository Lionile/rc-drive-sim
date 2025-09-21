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
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=None, allow_reverse=True):
        super(Actor, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        # Build MLP layers dynamically
        layers = []
        last_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            last_dim = h
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(last_dim, action_dim)

        self.max_action = max_action
        self.allow_reverse = allow_reverse

    def forward(self, state):
        a = state
        for layer in self.hidden_layers:
            a = F.relu(layer(a))
        action = self.max_action * torch.tanh(self.out(a))
        
        if not self.allow_reverse:
            # Scale from [-1, 1] to [0, 1] when reverse is not allowed
            action = (action + 1.0) / 2.0
        
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=None):
        super(Critic, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [400, 300]

        in_dim = state_dim + action_dim

        # Q1 architecture
        q1_layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            q1_layers.append(nn.Linear(last_dim, h))
            last_dim = h
        self.q1_hidden_layers = nn.ModuleList(q1_layers)
        self.q1_out = nn.Linear(last_dim, 1)

        # Q2 architecture (identical sizes)
        q2_layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            q2_layers.append(nn.Linear(last_dim, h))
            last_dim = h
        self.q2_hidden_layers = nn.ModuleList(q2_layers)
        self.q2_out = nn.Linear(last_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = sa
        for layer in self.q1_hidden_layers:
            q1 = F.relu(layer(q1))
        q1 = self.q1_out(q1)

        q2 = sa
        for layer in self.q2_hidden_layers:
            q2 = F.relu(layer(q2))
        q2 = self.q2_out(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = sa
        for layer in self.q1_hidden_layers:
            q1 = F.relu(layer(q1))
        q1 = self.q1_out(q1)
        return q1


class TD3Agent:
    def __init__(self, state_dim=3, action_dim=2, max_action=1.0, actor_lr=3e-4, critic_lr=3e-4,
                 actor_hidden_sizes=None, critic_hidden_sizes=None, allow_reverse=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim  # Store for checkpointing
        self.action_dim = action_dim

        # Store architecture
        self.actor_hidden_sizes = actor_hidden_sizes if actor_hidden_sizes is not None else [64, 64]
        self.critic_hidden_sizes = critic_hidden_sizes if critic_hidden_sizes is not None else [400, 300]

        self.actor = Actor(state_dim, action_dim, max_action, hidden_sizes=self.actor_hidden_sizes, allow_reverse=allow_reverse).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_sizes=self.actor_hidden_sizes, allow_reverse=allow_reverse).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_sizes=self.critic_hidden_sizes).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_sizes=self.critic_hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.max_action = max_action
        self.allow_reverse = allow_reverse
        self.training_mode = True
        
    def act(self, state, deterministic=False):
        """Get action from policy."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # NOTE: No exploration noise here. Exploration is handled in the Trainer
        # to avoid double-noise and to keep inference behavior clean.
        
        if not self.allow_reverse:
            # When reverse is not allowed, action is already in [0, 1] range
            return np.clip(action, 0.0, self.max_action)
        else:
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
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_action': self.max_action,
            'allow_reverse': self.allow_reverse,
            'actor_hidden_sizes': self.actor_hidden_sizes,
            'critic_hidden_sizes': self.critic_hidden_sizes,
        }, filename)
    
    def load(self, filename):
        """Load model parameters."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        # Verify state dimensions match
        if 'state_dim' in checkpoint and checkpoint['state_dim'] != self.state_dim:
            raise ValueError(f"Model state_dim mismatch: expected {self.state_dim}, got {checkpoint['state_dim']}")

        # Verify architecture compatibility when available
        ckpt_actor_sizes = checkpoint.get('actor_hidden_sizes', None)
        ckpt_critic_sizes = checkpoint.get('critic_hidden_sizes', None)
        if ckpt_actor_sizes is not None and ckpt_actor_sizes != self.actor_hidden_sizes:
            raise ValueError(f"Actor architecture mismatch: expected {self.actor_hidden_sizes}, got {ckpt_actor_sizes}. Construct agent with matching sizes.")
        if ckpt_critic_sizes is not None and ckpt_critic_sizes != self.critic_hidden_sizes:
            raise ValueError(f"Critic architecture mismatch: expected {self.critic_hidden_sizes}, got {ckpt_critic_sizes}. Construct agent with matching sizes.")
        
        # Load allow_reverse setting (default to True for backward compatibility)
        self.allow_reverse = checkpoint.get('allow_reverse', True)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


class TD3Controller(BaseController):
    def __init__(self, model_path=None, config=None):
        super().__init__()
        self.agent = None
        self.past_states_enabled = False
        self.past_states_buffer = None
        self.past_states_index = 0
        self.config = config  # Store config for past states setup
        
        if model_path:
            self.load_model(model_path)
    
    def act(self, observation, dt=1.0/60.0):
        """
        Get action from TD3 policy (deterministic for inference).
        
        Args:
            observation: Current sensor readings as numpy array
            dt: Time delta for physics update (not used by TD3)
            
        Returns:
            [left_wheel_velocity, right_wheel_velocity] in range [-1, 1] if allow_reverse=True, 
            or [0, 1] if allow_reverse=False
        """
        if self.agent is None:
            return [0.0, 0.0]
        
        # Ensure observation is numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Create augmented state if past states are enabled
        if self.past_states_enabled:
            augmented_obs = self._get_augmented_state(observation)
        else:
            augmented_obs = observation
        
        # Get deterministic action for inference
        action = self.agent.act(augmented_obs, deterministic=True)
        
        # Update past states buffer for next inference step
        if self.past_states_enabled:
            if hasattr(self, 'past_states_source') and self.past_states_source == 'wheels':
                self._update_past_states_buffer(action=action.copy())
            else:
                self._update_past_states_buffer(observation=observation)
        
        # Clamp to appropriate range based on allow_reverse setting
        if hasattr(self.agent, 'allow_reverse') and not self.agent.allow_reverse:
            action = np.clip(action, 0.0, 1.0)
        else:
            action = np.clip(action, -1.0, 1.0)
        
        return action.tolist()
    
    def load_model(self, model_path):
        """Load a trained TD3 model."""
        # Peek checkpoint to get state dimensions
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dim = checkpoint.get('state_dim', 3)  # Default to 3 for backward compatibility
        action_dim = checkpoint.get('action_dim', 2)
        max_action = checkpoint.get('max_action', 1.0)
        actor_hidden_sizes = checkpoint.get('actor_hidden_sizes', [64, 64])
        critic_hidden_sizes = checkpoint.get('critic_hidden_sizes', [400, 300])
        allow_reverse = checkpoint.get('allow_reverse', True)  # Default to True for backward compatibility
        
        # Create agent with correct dimensions
        self.agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                               actor_hidden_sizes=actor_hidden_sizes, critic_hidden_sizes=critic_hidden_sizes,
                               allow_reverse=allow_reverse)
        self.agent.load(model_path)
        self.agent.train_mode(False)  # Set to evaluation mode
        
        # Setup past states based on config or auto-detect
        base_state_dim = 3  # sensor readings
        if state_dim > base_state_dim:
            self.past_states_enabled = True
            
            # Use config if available, otherwise auto-detect
            if self.config and 'past_states' in self.config:
                past_config = self.config['past_states']
                self.past_states_source = past_config.get('source', 'wheels')
                self.past_states_count = past_config.get('count', 4)
                self.past_states_stride = past_config.get('stride', 3)
                print(f"✓ Using config past states: source={self.past_states_source}, count={self.past_states_count}, stride={self.past_states_stride}")
            else:
                # Fallback auto-detection (less reliable)
                past_dim = state_dim - base_state_dim
                if past_dim % 3 == 0:  # Likely sensors: 3 values per sensor state
                    self.past_states_count = past_dim // 3
                    self.past_states_source = 'sensors'
                elif past_dim % 2 == 0:  # Likely wheels: 2 values per wheel state
                    self.past_states_count = past_dim // 2
                    self.past_states_source = 'wheels'
                else:
                    # Fallback to wheels
                    self.past_states_count = past_dim // 2
                    self.past_states_source = 'wheels'
                self.past_states_stride = 3  # Default stride
                print(f"⚠ Auto-detected past states (less reliable): count={self.past_states_count}, source={self.past_states_source}")
            
            # Initialize ring buffer with correct dimensions
            buffer_size = self.past_states_stride * self.past_states_count + 1
            if self.past_states_source == 'wheels':
                self.past_states_buffer = np.zeros((buffer_size, 2), dtype=np.float32)  # [left_vel, right_vel]
            elif self.past_states_source == 'sensors':
                self.past_states_buffer = np.zeros((buffer_size, base_state_dim), dtype=np.float32)  # [sensor1, sensor2, sensor3]
        else:
            print("✓ No past states needed (state_dim == base_state_dim)")
        
        print(f"✓ Loaded TD3 model from {model_path} (state_dim={state_dim})")
    
    def _update_past_states_buffer(self, action=None, observation=None):
        """Update the ring buffer with new action or observation data."""
        if not self.past_states_enabled or self.past_states_buffer is None:
            return
        
        if hasattr(self, 'past_states_source') and self.past_states_source == 'wheels' and action is not None:
            # Store wheel velocities
            self.past_states_buffer[self.past_states_index] = action
        elif observation is not None:
            # Store sensor readings
            self.past_states_buffer[self.past_states_index] = observation
        
        # Advance ring buffer index
        self.past_states_index = (self.past_states_index + 1) % len(self.past_states_buffer)
    
    def _get_augmented_state(self, current_obs):
        """Create augmented state by combining current observation with past states."""
        if not self.past_states_enabled or self.past_states_buffer is None:
            return current_obs
        
        # Start with current observation
        augmented = [current_obs]
        
        # Add past states with stride
        for i in range(self.past_states_count):
            # Calculate index: go back by (i+1) * stride steps
            past_index = (self.past_states_index - (i + 1) * self.past_states_stride) % len(self.past_states_buffer)
            past_state = self.past_states_buffer[past_index]
            augmented.append(past_state)
        
        # Concatenate all states
        return np.concatenate(augmented, axis=0)
    
    def reset(self):
        """Reset controller state."""
        if self.past_states_enabled and self.past_states_buffer is not None:
            self.past_states_buffer.fill(0.0)
            self.past_states_index = 0


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
        
        # Separate noise parameters for exploration and target policy smoothing
        self.explore_sigma_start = config.get('explore_sigma_start', 0.2)
        self.explore_sigma_end = config.get('explore_sigma_end', 0.05)
        self.explore_clip = config.get('explore_clip', 0.5)
        self.smooth_sigma = config.get('smooth_sigma', 0.2)
        self.smooth_clip = config.get('smooth_clip', 0.5)
        
        # Exploration sigma decay (similar to epsilon decay)
        self.max_episodes = config.get('episodes', 1000)
        self.explore_sigma_decay = (self.explore_sigma_start - self.explore_sigma_end) / self.max_episodes if self.max_episodes > 0 else 0
        
        self.warmup_steps = config['warmup_steps']
        
        # Epsilon-greedy exploration decay
        self.epsilon_start = config.get('epsilon_start', 0.3)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.max_episodes = config.get('episodes', 1000)
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.max_episodes if self.max_episodes > 0 else 0
        
        # Past states configuration
        self.past_states_config = config.get('past_states', {})
        self.past_states_enabled = self.past_states_config.get('enabled', False)
        
        # Compute state dimension
        base_state_dim = 3  # Default sensor count
        if self.past_states_enabled:
            source = self.past_states_config.get('source', 'wheels')
            count = self.past_states_config.get('count', 4)
            if source == 'wheels':
                past_dim = count * 2  # 2 wheel velocities per past state
            elif source == 'sensors':
                past_dim = count * base_state_dim  # 3 sensors per past state
            else:
                raise ValueError(f"Unknown past_states source: {source}")
            state_dim = base_state_dim + past_dim
        else:
            state_dim = base_state_dim
        
        # Network architecture from config (with sensible defaults and backward compatibility)
        actor_hidden_sizes = config.get('actor_hidden_sizes', [64, 64])
        critic_hidden_sizes = config.get('critic_hidden_sizes', [400, 300])
        allow_reverse = config.get('allow_reverse', True)

        # Initialize agent
        self.agent = TD3Agent(state_dim=state_dim,
                              action_dim=2,
                              max_action=1.0,
                              actor_lr=config['actor_lr'],
                              critic_lr=config['critic_lr'],
                              actor_hidden_sizes=actor_hidden_sizes,
                              critic_hidden_sizes=critic_hidden_sizes,
                              allow_reverse=allow_reverse)
        
        # Training state
        self.total_steps = 0
        self.episode_num = 0
        
        # Past states ring buffer
        if self.past_states_enabled:
            source = self.past_states_config.get('source', 'wheels')
            count = self.past_states_config.get('count', 4)
            stride = self.past_states_config.get('stride', 3)
            
            # Ring buffer size needs to accommodate stride * count
            buffer_size = stride * count + 1  # +1 for current state
            if source == 'wheels':
                self.past_states_buffer = np.zeros((buffer_size, 2), dtype=np.float32)  # [left_vel, right_vel]
            elif source == 'sensors':
                self.past_states_buffer = np.zeros((buffer_size, base_state_dim), dtype=np.float32)
            
            self.past_states_source = source
            self.past_states_count = count
            self.past_states_stride = stride
            self.past_states_index = 0  # Current position in ring buffer
        else:
            self.past_states_buffer = None
            self.past_states_source = None
            self.past_states_count = 0
            self.past_states_stride = 0
            self.past_states_index = 0
        
        # Best model tracking
        self.best_reward = float('-inf')  # Track highest reward achieved
        self.recent_rewards = []  # Store recent episode rewards for rolling average calculation
        
        # Metrics tracking
        self.metrics = []

        # Periodic checkpointing
        self.save_every_episodes = int(config.get('save_every_episodes', 0) or 0)
        
        # Multi-map training support
        self.map_config = config.get('map', [])
        if isinstance(self.map_config, list):
            self.multi_map_enabled = True
            self.available_maps = self.map_config
            self.current_map_index = 0  # Track current position in cycle
            print(f"✓ Multi-map training enabled with {len(self.available_maps)} maps: {self.available_maps}")
        else:
            self.multi_map_enabled = False
            self.available_maps = [self.map_config]
            self.current_map_index = 0
            print(f"✓ Single map training: {self.map_config}")
        
        # Create per-map replay buffers
        self.replay_buffers = [ReplayMemory(config['buffer_size']) for _ in self.available_maps]
        self.num_maps = len(self.available_maps)
        
        # Graceful shutdown flag
        self.shutdown_requested = False
    
    def _select_next_map(self):
        """Select the next map in cyclic order for multi-map training."""
        if self.multi_map_enabled:
            # Get current bundle index from cycle
            bundle_index = self.current_map_index
            # Advance to next map in cycle
            self.current_map_index = (self.current_map_index + 1) % len(self.available_maps)
            return bundle_index
        else:
            return 0  # Single map case
        
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
            if self.shutdown_requested:
                print("Training stopped by user request")
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
            
            # Store episode reward for rolling average calculation
            self.recent_rewards.append(episode_reward)
            
            # Print progress
            if self.episode_num % 10 == 0:
                current_epsilon = self._get_current_epsilon()
                current_sigma = self._get_current_explore_sigma()
                print(f"Episode {self.episode_num}: Reward={episode_reward:.2f}, "
                      f"Steps={episode_steps}, Distance={episode_distance:.2f}, "
                      f"Avg Speed={avg_speed:.2f}, Epsilon={current_epsilon:.3f}, Sigma={current_sigma:.3f}")
            
            # Save best model if new best reward achieved
            self._save_best_model(episode_reward, self.episode_num)
            
            # Save metrics after every episode to avoid data loss
            self._save_metrics()
            
            # Periodic checkpoint save
            if self.save_every_episodes > 0 and (self.episode_num % self.save_every_episodes == 0):
                self._save_checkpoint(self.episode_num)
        
        # Save final model (metrics already saved after each episode)
        self._save_final_model()
        
        print(f"Training completed! Total episodes: {self.episode_num}, Total steps: {self.total_steps}")
    
    def _get_current_epsilon(self):
        """Calculate current epsilon value based on episode number (linear decay)."""
        if self.episode_num >= self.max_episodes:
            return self.epsilon_end
        current_epsilon = self.epsilon_start - (self.epsilon_decay * self.episode_num)
        return max(current_epsilon, self.epsilon_end)  # Ensure we don't go below epsilon_end
        
    def _get_current_explore_sigma(self):
        """Calculate current explore_sigma value based on episode number (linear decay)."""
        if self.episode_num >= self.max_episodes:
            return self.explore_sigma_end
        current_sigma = self.explore_sigma_start - (self.explore_sigma_decay * self.episode_num)
        return max(current_sigma, self.explore_sigma_end)  # Ensure we don't go below explore_sigma_end
        
    def _run_episode(self):
        """Run a single episode and collect experience."""
        # Select next map for multi-map training (cyclic order)
        if self.multi_map_enabled:
            selected_map = self._select_next_map()
            self.env.set_map(selected_map)
            if self.episode_num % 10 == 0:  # Log map selection every 10 episodes
                print(f"Episode {self.episode_num + 1}: Using map {selected_map}")
        else:
            selected_map = 0  # Single map case
        
        obs = self.env.reset()
        self._reset_past_states_buffer()  # Reset past states buffer at episode start
        episode_reward = 0
        episode_steps = 0
        episode_distance = 0
        prev_pos = None
        epsilon = self._get_current_epsilon()  # Linear decay from epsilon_start to epsilon_end

        while True:
            # Create augmented state for policy input
            augmented_obs = self._get_augmented_state(obs)
            
            # Get action
            if self.total_steps < self.warmup_steps:
                # Random actions during warmup
                if not self.agent.allow_reverse:
                    action = np.random.uniform(0, 1, 2)
                else:
                    action = np.random.uniform(-1, 1, 2)
            else:
                # Epsilon-greedy: sometimes take a random action to encourage turning
                if np.random.rand() < epsilon:
                    if not self.agent.allow_reverse:
                        action = np.random.uniform(0, 1, 2)
                    else:
                        action = np.random.uniform(-1, 1, 2)
                else:
                    # Policy action (deterministic), then add Gaussian exploration noise
                    action = self.agent.act(augmented_obs, deterministic=True)
                    current_sigma = self._get_current_explore_sigma()
                    noise = np.random.normal(0, current_sigma, size=action.shape)
                    noise = np.clip(noise, -self.explore_clip, self.explore_clip)
                    if not self.agent.allow_reverse:
                        # When reverse is not allowed, action is in [0, 1]
                        action = np.clip(action + noise, 0, 1)
                    else:
                        action = np.clip(action + noise, -1, 1)

            # Update past states buffer with current action or observation
            if self.past_states_source == 'wheels':
                self._update_past_states_buffer(action=action)
            elif self.past_states_source == 'sensors':
                self._update_past_states_buffer(observation=obs)
            
            # Step environment
            dt = 1.0 / self.config.get('physics_fps', 60)  # Use configurable physics frequency
            next_obs, reward, terminated, truncated, info = self.env.step(action, dt=dt)
            
            # Create augmented next state
            augmented_next_obs = self._get_augmented_state(next_obs)
            
            # Calculate distance traveled
            current_pos = info['position']
            if prev_pos is not None:
                episode_distance += distance(prev_pos, current_pos)
            prev_pos = current_pos
            
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Store transition in replay buffer (use augmented states)
            done = terminated or truncated
            self.replay_buffers[selected_map].push(
                torch.FloatTensor(augmented_obs),
                torch.FloatTensor(action),
                torch.FloatTensor(augmented_next_obs),
                torch.FloatTensor([reward]),
                torch.FloatTensor([float(done)])
            )
            
            # Train agent if enough experience and past warmup
            total_samples = sum(len(buffer) for buffer in self.replay_buffers)
            if total_samples > self.batch_size and self.total_steps > self.warmup_steps:
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
                        elif event.key == pygame.K_d:
                            self.env.show_distance_heatmap = not self.env.show_distance_heatmap
                            print(f"Distance heatmap: {'ON' if self.env.show_distance_heatmap else 'OFF'}")
                        elif event.key == pygame.K_h:
                            self.render = not self.render
                            print(f"Rendering: {'ON' if self.render else 'OFF'}")
                        elif event.key == pygame.K_q:
                            print("Graceful shutdown requested (Q pressed)...")
                            self.shutdown_requested = True
                            return episode_reward, episode_steps, episode_distance  # Exit current episode
                
                # Only render if currently enabled
                if self.render:
                    self.env.render()
                    pygame.display.flip()
                    # Control frame rate during rendering
                    physics_fps = self.config.get('physics_fps', 60)
                    pygame.time.wait(int(1000 / physics_fps))  # Match physics FPS
            
            obs = next_obs
            
            if done:
                break
        
        return episode_reward, episode_steps, episode_distance
    
    def _update_agent(self):
        """Update TD3 agent with a batch of experience."""
        # Sample proportionally from each map's replay buffer
        samples_per_buffer = round(self.batch_size / self.num_maps)
        
        all_state_batches = []
        all_action_batches = []
        all_reward_batches = []
        all_next_state_batches = []
        all_done_batches = []
        
        for buffer in self.replay_buffers:
            if len(buffer) >= samples_per_buffer:
                # Sample the full amount from this buffer
                sample_size = samples_per_buffer
            else:
                # Sample all available from this buffer
                sample_size = len(buffer)
            
            if sample_size > 0:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = buffer.sample(sample_size)
                all_state_batches.append(state_batch)
                all_action_batches.append(action_batch)
                all_reward_batches.append(reward_batch)
                all_next_state_batches.append(next_state_batch)
                all_done_batches.append(done_batch)
        
        # Concatenate all samples
        if all_state_batches:
            state_batch = torch.cat(all_state_batches, dim=0)
            action_batch = torch.cat(all_action_batches, dim=0)
            reward_batch = torch.cat(all_reward_batches, dim=0)
            next_state_batch = torch.cat(all_next_state_batches, dim=0)
            done_batch = torch.cat(all_done_batches, dim=0)
        else:
            # Fallback if no samples available (shouldn't happen in practice)
            return
        
        state_batch = state_batch.to(self.agent.device)
        action_batch = action_batch.to(self.agent.device)
        reward_batch = reward_batch.to(self.agent.device)
        next_state_batch = next_state_batch.to(self.agent.device)
        done_batch = done_batch.to(self.agent.device)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action_batch) * self.smooth_sigma).clamp(-self.smooth_clip, self.smooth_clip)
            next_action = self.agent.actor_target(next_state_batch) + noise
            if self.agent.allow_reverse:
                next_action = next_action.clamp(-self.agent.max_action, self.agent.max_action)
            else:
                next_action = next_action.clamp(0.0, 1.0)
            
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
    
    def _update_past_states_buffer(self, action=None, observation=None):
        """Update the ring buffer with new action or observation data."""
        if not self.past_states_enabled:
            return
        
        if self.past_states_source == 'wheels' and action is not None:
            # Store wheel velocities
            self.past_states_buffer[self.past_states_index] = action
        elif self.past_states_source == 'sensors' and observation is not None:
            # Store sensor readings
            self.past_states_buffer[self.past_states_index] = observation
        
        # Advance ring buffer index
        self.past_states_index = (self.past_states_index + 1) % len(self.past_states_buffer)
    
    def _get_augmented_state(self, current_obs):
        """Create augmented state by combining current observation with past states."""
        if not self.past_states_enabled:
            return current_obs
        
        # Start with current observation
        augmented = [current_obs]
        
        # Add past states with stride
        for i in range(self.past_states_count):
            # Calculate index: go back by (i+1) * stride steps
            past_index = (self.past_states_index - (i + 1) * self.past_states_stride) % len(self.past_states_buffer)
            past_state = self.past_states_buffer[past_index]
            augmented.append(past_state)
        
        # Concatenate all states
        return np.concatenate(augmented, axis=0)
    
    def _reset_past_states_buffer(self):
        """Reset past states buffer (call at episode start)."""
        if self.past_states_enabled:
            self.past_states_buffer.fill(0.0)
            self.past_states_index = 0
    
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
        """Save model if it achieves a new best reward (or rolling average for multi-map training)."""
        if self.multi_map_enabled and len(self.recent_rewards) >= self.num_maps:
            # For multi-map training, use rolling average of last n episodes
            rolling_avg = sum(self.recent_rewards[-self.num_maps:]) / self.num_maps
            if rolling_avg > self.best_reward:
                self.best_reward = rolling_avg
                best_path = self.save_dir / "best_model.pt"
                self.agent.save(best_path)
                print(f"✓ New best model saved at episode {episode_num} with rolling avg reward {rolling_avg:.2f} to {best_path}")
                return True
        elif not self.multi_map_enabled:
            # For single-map training, use individual episode reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                best_path = self.save_dir / "best_model.pt"
                self.agent.save(best_path)
                print(f"✓ New best model saved at episode {episode_num} with reward {episode_reward:.2f} to {best_path}")
                return True
        return False
    
    def _save_metrics(self):
        """Save training metrics to CSV (overwrites previous file)."""
        metrics_path = self.save_dir / "metrics.csv"
        
        with open(metrics_path, 'w', newline='') as csvfile:
            fieldnames = ['episode', 'total_reward', 'steps', 'total_distance', 'avg_speed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metrics in self.metrics:
                writer.writerow(metrics)
        
        # Only print confirmation periodically to avoid spam
        if self.episode_num % 50 == 0 or self.shutdown_requested:
            print(f"✓ Metrics saved to {metrics_path}")
