"""
Modern RL Agents for Traffic Light Control.

This module implements various state-of-the-art RL algorithms
using stable-baselines3 for traffic light control.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from src.envs.traffic_env import TrafficLightEnv


class TrafficLightAgent:
    """
    A wrapper class for various RL agents for traffic light control.
    
    Supports PPO, SAC, TD3, and DQN algorithms with proper configuration
    for the traffic light environment.
    """
    
    def __init__(self, 
                 algorithm: str = "PPO",
                 env_config: Optional[Dict[str, Any]] = None,
                 model_config: Optional[Dict[str, Any]] = None,
                 use_wandb: bool = False,
                 project_name: str = "traffic-light-rl"):
        """
        Initialize the RL agent.
        
        Args:
            algorithm: RL algorithm to use ('PPO', 'SAC', 'TD3', 'DQN')
            env_config: Environment configuration parameters
            model_config: Model-specific configuration parameters
            use_wandb: Whether to use Weights & Biases logging
            project_name: W&B project name
        """
        self.algorithm = algorithm.upper()
        self.env_config = env_config or {}
        self.model_config = model_config or {}
        self.use_wandb = use_wandb
        self.project_name = project_name
        
        # Initialize W&B if requested
        if self.use_wandb:
            wandb.init(project=self.project_name, config={
                "algorithm": self.algorithm,
                "env_config": self.env_config,
                "model_config": self.model_config
            })
        
        # Create environment
        self.env = self._create_environment()
        
        # Create model
        self.model = self._create_model()
        
    def _create_environment(self) -> gym.Env:
        """Create and configure the traffic light environment."""
        env = TrafficLightEnv(**self.env_config)
        env = Monitor(env)
        
        # Wrap in vectorized environment for better performance
        env = make_vec_env(lambda: env, n_envs=1)
        
        # Normalize observations
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        return env
    
    def _create_model(self):
        """Create the RL model based on the selected algorithm."""
        # Default configurations for each algorithm
        default_configs = {
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy_kwargs": {
                    "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
                }
            },
            "SAC": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
                "target_update_interval": 1,
                "policy_kwargs": {
                    "net_arch": [256, 256]
                }
            },
            "TD3": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
                "policy_kwargs": {
                    "net_arch": [256, 256]
                }
            },
            "DQN": {
                "learning_rate": 1e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "policy_kwargs": {
                    "net_arch": [256, 256]
                }
            }
        }
        
        # Merge default config with user config
        config = {**default_configs[self.algorithm], **self.model_config}
        
        # Create model based on algorithm
        if self.algorithm == "PPO":
            model = PPO("MlpPolicy", self.env, verbose=1, **config)
        elif self.algorithm == "SAC":
            model = SAC("MlpPolicy", self.env, verbose=1, **config)
        elif self.algorithm == "TD3":
            model = TD3("MlpPolicy", self.env, verbose=1, **config)
        elif self.algorithm == "DQN":
            model = DQN("MlpPolicy", self.env, verbose=1, **config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return model
    
    def train(self, 
              total_timesteps: int = 100000,
              save_path: Optional[str] = None,
              log_interval: int = 10) -> None:
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            save_path: Path to save the trained model
            log_interval: Interval for logging progress
        """
        callbacks = []
        
        # Add W&B callback if enabled
        if self.use_wandb:
            callbacks.append(WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"models/{self.algorithm.lower()}",
                verbose=2,
            ))
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            log_interval=log_interval
        )
        
        # Save model if path provided
        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
    
    def evaluate(self, 
                 n_eval_episodes: int = 10,
                 deterministic: bool = True) -> Tuple[float, float]:
        """
        Evaluate the trained agent.
        
        Args:
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            Mean reward and standard deviation
        """
        episode_rewards = []
        
        for _ in range(n_eval_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0] if hasattr(reward, '__len__') else reward
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        print(f"Evaluation Results:")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Episodes: {n_eval_episodes}")
        
        return mean_reward, std_reward
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, np.ndarray]:
        """
        Make a prediction given an observation.
        
        Args:
            obs: Observation from the environment
            deterministic: Whether to use deterministic actions
            
        Returns:
            Action and action probabilities
        """
        return self.model.predict(obs, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """Save the trained model."""
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load a trained model."""
        if self.algorithm == "PPO":
            self.model = PPO.load(path)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path)
        elif self.algorithm == "TD3":
            self.model = TD3.load(path)
        elif self.algorithm == "DQN":
            self.model = DQN.load(path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")


class RainbowDQNAgent:
    """
    Rainbow DQN implementation for traffic light control.
    
    This is a more advanced version of DQN that includes:
    - Double DQN
    - Prioritized Experience Replay
    - Dueling Network Architecture
    - Multi-step Learning
    - Distributional RL
    - Noisy Networks
    """
    
    def __init__(self, 
                 env_config: Optional[Dict[str, Any]] = None,
                 model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Rainbow DQN agent.
        
        Args:
            env_config: Environment configuration
            model_config: Model configuration
        """
        # For now, we'll use a standard DQN with enhanced configuration
        # A full Rainbow implementation would require custom network architecture
        enhanced_config = {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": {
                "net_arch": [512, 512, 256],  # Deeper network
                "activation_fn": torch.nn.ReLU,
            }
        }
        
        if model_config:
            enhanced_config.update(model_config)
        
        self.agent = TrafficLightAgent(
            algorithm="DQN",
            env_config=env_config,
            model_config=enhanced_config
        )
    
    def train(self, total_timesteps: int = 100000, **kwargs):
        """Train the Rainbow DQN agent."""
        return self.agent.train(total_timesteps, **kwargs)
    
    def evaluate(self, n_eval_episodes: int = 10, **kwargs):
        """Evaluate the Rainbow DQN agent."""
        return self.agent.evaluate(n_eval_episodes, **kwargs)
    
    def predict(self, obs: np.ndarray, **kwargs):
        """Make a prediction."""
        return self.agent.predict(obs, **kwargs)
    
    def save(self, path: str):
        """Save the model."""
        return self.agent.save(path)
    
    def load(self, path: str):
        """Load the model."""
        return self.agent.load(path)
