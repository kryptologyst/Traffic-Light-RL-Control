#!/usr/bin/env python3
"""
Main training script for Traffic Light RL project.

This script provides a command-line interface for training and evaluating
various RL algorithms on the traffic light control environment.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.modern_agents import TrafficLightAgent, RainbowDQNAgent
from src.utils.config import Config
from src.utils.visualization import RLVisualizer


def train_agent(algorithm: str, 
                config: Config, 
                total_timesteps: int = None,
                save_path: str = None,
                use_wandb: bool = False) -> TrafficLightAgent:
    """
    Train an RL agent.
    
    Args:
        algorithm: RL algorithm to use
        config: Configuration object
        total_timesteps: Number of timesteps to train
        save_path: Path to save the model
        use_wandb: Whether to use W&B logging
        
    Returns:
        Trained agent
    """
    print(f"Training {algorithm} agent...")
    
    # Get configurations
    env_config = config.get_environment_config()
    model_config = config.get_model_config(algorithm)
    training_config = config.get_training_config()
    logging_config = config.get_logging_config()
    
    # Override timesteps if provided
    if total_timesteps is None:
        total_timesteps = training_config.get("total_timesteps", 100000)
    
    # Create agent
    if algorithm.upper() == "RAINBOW":
        agent = RainbowDQNAgent(env_config, model_config)
    else:
        agent = TrafficLightAgent(
            algorithm=algorithm,
            env_config=env_config,
            model_config=model_config,
            use_wandb=use_wandb or logging_config.get("use_wandb", False),
            project_name=logging_config.get("project_name", "traffic-light-rl")
        )
    
    # Train the agent
    agent.train(
        total_timesteps=total_timesteps,
        save_path=save_path,
        log_interval=training_config.get("log_interval", 10)
    )
    
    return agent


def evaluate_agent(agent: TrafficLightAgent, 
                  config: Config,
                  n_eval_episodes: int = None) -> Dict[str, float]:
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained agent
        config: Configuration object
        n_eval_episodes: Number of episodes to evaluate
        
    Returns:
        Evaluation metrics
    """
    print(f"Evaluating {agent.algorithm} agent...")
    
    training_config = config.get_training_config()
    if n_eval_episodes is None:
        n_eval_episodes = training_config.get("n_eval_episodes", 10)
    
    mean_reward, std_reward = agent.evaluate(n_eval_episodes=n_eval_episodes)
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_episodes": n_eval_episodes
    }


def compare_algorithms(algorithms: List[str], 
                      config: Config,
                      total_timesteps: int = None,
                      n_eval_episodes: int = None) -> Dict[str, Any]:
    """
    Compare multiple algorithms.
    
    Args:
        algorithms: List of algorithms to compare
        config: Configuration object
        total_timesteps: Number of timesteps to train each algorithm
        n_eval_episodes: Number of episodes for evaluation
        
    Returns:
        Comparison results
    """
    print(f"Comparing algorithms: {', '.join(algorithms)}")
    
    results = {
        "training_rewards": {},
        "evaluation_metrics": {}
    }
    
    for algorithm in algorithms:
        print(f"\n{'='*50}")
        print(f"Training {algorithm}")
        print(f"{'='*50}")
        
        # Train agent
        agent = train_agent(algorithm, config, total_timesteps)
        
        # Evaluate agent
        eval_metrics = evaluate_agent(agent, config, n_eval_episodes)
        results["evaluation_metrics"][algorithm] = eval_metrics
        
        # Note: For training rewards, we would need to modify the agent
        # to return episode rewards during training. For now, we'll use
        # evaluation metrics as a proxy.
        results["training_rewards"][algorithm] = [eval_metrics["mean_reward"]]
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Traffic Light RL Training")
    
    # Algorithm selection
    parser.add_argument("--algorithm", "-a", 
                       choices=["PPO", "SAC", "TD3", "DQN", "RAINBOW"],
                       default="PPO",
                       help="RL algorithm to use")
    
    # Training parameters
    parser.add_argument("--timesteps", "-t", type=int, default=100000,
                       help="Total timesteps for training")
    parser.add_argument("--eval-episodes", "-e", type=int, default=10,
                       help="Number of episodes for evaluation")
    
    # Configuration
    parser.add_argument("--config", "-c", type=str, default=None,
                       help="Path to configuration file")
    
    # Output
    parser.add_argument("--save-path", "-s", type=str, default=None,
                       help="Path to save trained model")
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                       help="Output directory for results")
    
    # Logging
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot generation")
    
    # Comparison mode
    parser.add_argument("--compare", nargs="+", 
                       choices=["PPO", "SAC", "TD3", "DQN", "RAINBOW"],
                       help="Compare multiple algorithms")
    
    # Load model
    parser.add_argument("--load-model", "-l", type=str, default=None,
                       help="Path to load pre-trained model")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = Config(args.config)
    config.create_directories()
    
    # Initialize visualizer
    visualizer = RLVisualizer(save_dir=os.path.join(args.output_dir, "plots"))
    
    if args.compare:
        # Comparison mode
        results = compare_algorithms(
            args.compare, 
            config, 
            args.timesteps, 
            args.eval_episodes
        )
        
        if not args.no_plots:
            # Plot comparison results
            visualizer.plot_algorithm_comparison(
                results["training_rewards"], 
                save=True, 
                show=False
            )
            visualizer.plot_evaluation_results(
                results["evaluation_metrics"], 
                save=True, 
                show=False
            )
            visualizer.create_summary_report(
                results["training_rewards"],
                results["evaluation_metrics"],
                save=True
            )
        
        print("\nComparison Results:")
        for algorithm, metrics in results["evaluation_metrics"].items():
            print(f"{algorithm}: Mean Reward = {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    else:
        # Single algorithm mode
        if args.load_model:
            # Load pre-trained model
            print(f"Loading model from {args.load_model}")
            agent = TrafficLightAgent(args.algorithm)
            agent.load(args.load_model)
        else:
            # Train new model
            save_path = args.save_path or os.path.join(args.output_dir, f"{args.algorithm.lower()}_model")
            agent = train_agent(
                args.algorithm, 
                config, 
                args.timesteps, 
                save_path,
                args.wandb
            )
        
        # Evaluate the agent
        eval_metrics = evaluate_agent(agent, config, args.eval_episodes)
        
        print(f"\nFinal Results for {args.algorithm}:")
        print(f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        
        if not args.no_plots:
            # Create plots (using evaluation metrics as proxy for training rewards)
            training_rewards = [eval_metrics['mean_reward']] * 100  # Placeholder
            visualizer.plot_training_curves(
                training_rewards, 
                args.algorithm, 
                save=True, 
                show=False
            )


if __name__ == "__main__":
    main()
