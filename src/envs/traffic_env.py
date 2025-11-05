"""
Modern Traffic Light Control Environment using Gymnasium interface.

This module implements a traffic intersection simulation where an RL agent
learns to control traffic lights to minimize vehicle waiting times.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class TrafficLightEnv(gym.Env):
    """
    A traffic light control environment for reinforcement learning.
    
    The environment simulates a simple 2-way intersection (North-South and East-West)
    where vehicles arrive randomly and an agent controls traffic light phases
    to minimize total waiting time.
    
    Observation Space:
        - ns_queue: Number of vehicles waiting in North-South direction (0-10)
        - ew_queue: Number of vehicles waiting in East-West direction (0-10)  
        - light_state: Current traffic light state (0=NS green, 1=EW green)
        
    Action Space:
        - 0: Keep current light state
        - 1: Switch to opposite light state
        
    Reward:
        Negative of total queue length (minimize waiting time)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 max_queue: int = 10,
                 arrival_prob: float = 0.6,
                 max_vehicles_per_step: int = 2,
                 max_steps: int = 100,
                 render_mode: Optional[str] = None):
        """
        Initialize the traffic light environment.
        
        Args:
            max_queue: Maximum number of vehicles per direction
            arrival_prob: Probability of vehicle arrival per step
            max_vehicles_per_step: Maximum vehicles that can pass per step
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.max_queue = max_queue
        self.arrival_prob = arrival_prob
        self.max_vehicles_per_step = max_vehicles_per_step
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            "ns_queue": spaces.Discrete(max_queue + 1),
            "ew_queue": spaces.Discrete(max_queue + 1),
            "light_state": spaces.Discrete(2)
        })
        
        self.action_space = spaces.Discrete(2)
        
        # Initialize state
        self.ns_queue = 0
        self.ew_queue = 0
        self.light_state = 0  # 0: NS green, 1: EW green
        self.step_count = 0
        
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Initialize with random queue lengths
        self.ns_queue = random.randint(0, min(5, self.max_queue))
        self.ew_queue = random.randint(0, min(5, self.max_queue))
        self.light_state = random.randint(0, 1)
        self.step_count = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, int], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0 or 1.")
            
        # Switch light if action is 1
        if action == 1:
            self.light_state = 1 - self.light_state
            
        # Vehicles pass based on green light
        if self.light_state == 0:  # NS green
            passed = min(self.max_vehicles_per_step, self.ns_queue)
            self.ns_queue -= passed
        else:  # EW green
            passed = min(self.max_vehicles_per_step, self.ew_queue)
            self.ew_queue -= passed
            
        # Random vehicle arrivals
        self.ns_queue += np.random.binomial(1, self.arrival_prob)
        self.ew_queue += np.random.binomial(1, self.arrival_prob)
        
        # Limit queue sizes
        self.ns_queue = min(self.ns_queue, self.max_queue)
        self.ew_queue = min(self.ew_queue, self.max_queue)
        
        # Calculate reward (negative total queue length)
        reward = -(self.ns_queue + self.ew_queue)
        
        # Check termination conditions
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False  # No early termination in this environment
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, int]:
        """Get current observation."""
        return {
            "ns_queue": self.ns_queue,
            "ew_queue": self.ew_queue,
            "light_state": self.light_state
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        return {
            "step_count": self.step_count,
            "total_queue": self.ns_queue + self.ew_queue,
            "light_phase": "NS Green" if self.light_state == 0 else "EW Green"
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step {self.step_count}: NS={self.ns_queue}, EW={self.ew_queue}, "
                  f"Light={'NS' if self.light_state == 0 else 'EW'}")
        elif self.render_mode == "rgb_array":
            # Could implement visual rendering here
            pass
    
    def close(self):
        """Clean up resources."""
        pass


# Register the environment with gymnasium
gym.register(
    id="TrafficLight-v1",
    entry_point="src.envs.traffic_env:TrafficLightEnv",
    max_episode_steps=100,
)
