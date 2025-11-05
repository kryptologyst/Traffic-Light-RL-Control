"""
Configuration management utilities for the Traffic Light RL project.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the Traffic Light RL project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "default_config.yaml"
        
        self.config_path = str(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            print("Using default configuration...")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            print("Using default configuration...")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "environment": {
                "max_queue": 10,
                "arrival_prob": 0.6,
                "max_vehicles_per_step": 2,
                "max_steps": 100,
                "render_mode": None
            },
            "training": {
                "total_timesteps": 100000,
                "log_interval": 10,
                "save_interval": 10000,
                "eval_interval": 5000,
                "n_eval_episodes": 10
            },
            "logging": {
                "use_wandb": False,
                "use_tensorboard": True,
                "project_name": "traffic-light-rl",
                "log_dir": "logs",
                "save_dir": "models"
            },
            "paths": {
                "models_dir": "models",
                "logs_dir": "logs",
                "configs_dir": "configs",
                "data_dir": "data"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'environment.max_queue')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.get("environment", {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get("training", {})
    
    def get_model_config(self, algorithm: str) -> Dict[str, Any]:
        """Get model configuration for specific algorithm."""
        return self.get(f"models.{algorithm}", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.get("paths", {})
    
    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'environment.max_queue')
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        if path is None:
            path = self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        paths_config = self.get_paths_config()
        
        for key, path in paths_config.items():
            if isinstance(path, str):
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self.update(key, value)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(config_path='{self.config_path}')"
