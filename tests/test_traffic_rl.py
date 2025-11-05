"""
Unit tests for the Traffic Light RL project.

This module contains comprehensive tests for all major components
including environment, agents, and utilities.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.envs.traffic_env import TrafficLightEnv
from src.agents.modern_agents import TrafficLightAgent, RainbowDQNAgent
from src.utils.config import Config
from src.utils.visualization import RLVisualizer


class TestTrafficLightEnv:
    """Test cases for TrafficLightEnv."""
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        env = TrafficLightEnv()
        
        assert env.max_queue == 10
        assert env.arrival_prob == 0.6
        assert env.max_vehicles_per_step == 2
        assert env.max_steps == 100
        
        # Check observation space
        assert env.observation_space is not None
        assert env.action_space is not None
    
    def test_reset(self):
        """Test environment reset."""
        env = TrafficLightEnv()
        obs, info = env.reset(seed=42)
        
        # Check observation structure
        assert isinstance(obs, dict)
        assert "ns_queue" in obs
        assert "ew_queue" in obs
        assert "light_state" in obs
        
        # Check observation bounds
        assert 0 <= obs["ns_queue"] <= env.max_queue
        assert 0 <= obs["ew_queue"] <= env.max_queue
        assert obs["light_state"] in [0, 1]
        
        # Check info structure
        assert isinstance(info, dict)
        assert "step_count" in info
        assert info["step_count"] == 0
    
    def test_step(self):
        """Test environment step."""
        env = TrafficLightEnv()
        obs, _ = env.reset(seed=42)
        
        # Test valid action
        action = 0
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Check return types
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check observation bounds
        assert 0 <= next_obs["ns_queue"] <= env.max_queue
        assert 0 <= next_obs["ew_queue"] <= env.max_queue
        assert next_obs["light_state"] in [0, 1]
        
        # Check reward is negative (we want to minimize queue)
        assert reward <= 0
    
    def test_invalid_action(self):
        """Test invalid action handling."""
        env = TrafficLightEnv()
        obs, _ = env.reset()
        
        # Test invalid action
        with pytest.raises(ValueError):
            env.step(2)  # Invalid action
    
    def test_light_switching(self):
        """Test traffic light switching."""
        env = TrafficLightEnv()
        obs, _ = env.reset(seed=42)
        
        initial_light = obs["light_state"]
        
        # Switch light
        next_obs, _, _, _, _ = env.step(1)
        
        assert next_obs["light_state"] == 1 - initial_light
    
    def test_episode_termination(self):
        """Test episode termination."""
        env = TrafficLightEnv(max_steps=5)
        obs, _ = env.reset()
        
        terminated = False
        step_count = 0
        
        while not terminated and step_count < 10:
            obs, _, terminated, _, _ = env.step(0)
            step_count += 1
        
        assert terminated
        assert step_count == 5


class TestTrafficLightAgent:
    """Test cases for TrafficLightAgent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = TrafficLightAgent(algorithm="PPO")
        
        assert agent.algorithm == "PPO"
        assert agent.env is not None
        assert agent.model is not None
    
    def test_unsupported_algorithm(self):
        """Test unsupported algorithm handling."""
        with pytest.raises(ValueError):
            TrafficLightAgent(algorithm="INVALID")
    
    def test_predict(self):
        """Test agent prediction."""
        agent = TrafficLightAgent(algorithm="PPO")
        
        # Create dummy observation
        obs = {"ns_queue": 5, "ew_queue": 3, "light_state": 0}
        
        action, _ = agent.predict(obs)
        
        assert action in [0, 1]
    
    def test_save_load(self):
        """Test model saving and loading."""
        agent = TrafficLightAgent(algorithm="PPO")
        
        # Test save
        save_path = "test_model"
        agent.save(save_path)
        
        assert os.path.exists(f"{save_path}.zip")
        
        # Test load
        agent.load(save_path)
        
        # Clean up
        os.remove(f"{save_path}.zip")


class TestRainbowDQNAgent:
    """Test cases for RainbowDQNAgent."""
    
    def test_rainbow_initialization(self):
        """Test Rainbow DQN agent initialization."""
        agent = RainbowDQNAgent()
        
        assert agent.agent is not None
        assert agent.agent.algorithm == "DQN"
    
    def test_rainbow_methods(self):
        """Test Rainbow DQN agent methods."""
        agent = RainbowDQNAgent()
        
        # Test that all methods exist and are callable
        assert callable(agent.train)
        assert callable(agent.evaluate)
        assert callable(agent.predict)
        assert callable(agent.save)
        assert callable(agent.load)


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_initialization(self):
        """Test config initialization."""
        config = Config()
        
        assert isinstance(config.config, dict)
        assert "environment" in config.config
        assert "training" in config.config
    
    def test_get_method(self):
        """Test config get method."""
        config = Config()
        
        # Test simple key
        env_config = config.get("environment")
        assert isinstance(env_config, dict)
        
        # Test dot notation
        max_queue = config.get("environment.max_queue")
        assert isinstance(max_queue, int)
        
        # Test default value
        non_existent = config.get("non.existent.key", "default")
        assert non_existent == "default"
    
    def test_update_method(self):
        """Test config update method."""
        config = Config()
        
        # Test simple update
        config.update("test_key", "test_value")
        assert config.get("test_key") == "test_value"
        
        # Test dot notation update
        config.update("environment.test_param", 42)
        assert config.get("environment.test_param") == 42
    
    def test_get_specific_configs(self):
        """Test getting specific configuration sections."""
        config = Config()
        
        env_config = config.get_environment_config()
        assert isinstance(env_config, dict)
        
        training_config = config.get_training_config()
        assert isinstance(training_config, dict)
        
        model_config = config.get_model_config("PPO")
        assert isinstance(model_config, dict)
        
        logging_config = config.get_logging_config()
        assert isinstance(logging_config, dict)
        
        paths_config = config.get_paths_config()
        assert isinstance(paths_config, dict)


class TestRLVisualizer:
    """Test cases for RLVisualizer."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = RLVisualizer()
        
        assert visualizer.save_dir == "plots"
        assert os.path.exists(visualizer.save_dir)
    
    def test_plot_training_curves(self):
        """Test training curves plotting."""
        visualizer = RLVisualizer()
        
        # Generate dummy data
        rewards = np.random.normal(-5, 2, 1000)
        
        # Test plotting (without showing)
        visualizer.plot_training_curves(
            rewards, 
            algorithm="Test", 
            save=False, 
            show=False
        )
    
    def test_plot_algorithm_comparison(self):
        """Test algorithm comparison plotting."""
        visualizer = RLVisualizer()
        
        # Generate dummy data
        results = {
            "PPO": np.random.normal(-5, 2, 1000),
            "SAC": np.random.normal(-6, 2, 1000),
            "DQN": np.random.normal(-7, 2, 1000)
        }
        
        # Test plotting (without showing)
        visualizer.plot_algorithm_comparison(
            results, 
            save=False, 
            show=False
        )
    
    def test_plot_evaluation_results(self):
        """Test evaluation results plotting."""
        visualizer = RLVisualizer()
        
        # Generate dummy data
        results = {
            "PPO": {"mean_reward": -5.2, "std_reward": 1.5},
            "SAC": {"mean_reward": -6.1, "std_reward": 1.8},
            "DQN": {"mean_reward": -7.3, "std_reward": 2.1}
        }
        
        # Test plotting (without showing)
        visualizer.plot_evaluation_results(
            results, 
            save=False, 
            show=False
        )


class TestIntegration:
    """Integration tests."""
    
    def test_full_training_cycle(self):
        """Test complete training cycle."""
        # Create environment
        env = TrafficLightEnv(max_steps=10)
        
        # Create agent
        agent = TrafficLightAgent(algorithm="PPO")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = agent.predict(obs)
        
        assert action in [0, 1]
        
        # Test step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_config_with_agent(self):
        """Test configuration with agent."""
        config = Config()
        
        # Update config
        config.update("environment.max_queue", 15)
        config.update("training.total_timesteps", 1000)
        
        # Create agent with config
        env_config = config.get_environment_config()
        model_config = config.get_model_config("PPO")
        
        agent = TrafficLightAgent(
            algorithm="PPO",
            env_config=env_config,
            model_config=model_config
        )
        
        assert agent.env_config["max_queue"] == 15


# Fixtures for pytest
@pytest.fixture
def sample_env():
    """Sample environment for testing."""
    return TrafficLightEnv(max_steps=10)


@pytest.fixture
def sample_agent():
    """Sample agent for testing."""
    return TrafficLightAgent(algorithm="PPO")


@pytest.fixture
def sample_config():
    """Sample config for testing."""
    return Config()


if __name__ == "__main__":
    pytest.main([__file__])
