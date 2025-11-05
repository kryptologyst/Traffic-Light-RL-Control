#!/usr/bin/env python3
"""
Demo script for Traffic Light RL project.

This script demonstrates the basic functionality of the project
with a simple training and evaluation example.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.envs.traffic_env import TrafficLightEnv
from src.agents.modern_agents import TrafficLightAgent
from src.utils.config import Config
from src.utils.visualization import RLVisualizer


def main():
    """Main demo function."""
    print("🚦 Traffic Light RL Control - Demo")
    print("=" * 50)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = Config()
    config.create_directories()
    print("✅ Configuration loaded and directories created")
    
    # Create environment
    print("\n2. Creating environment...")
    env = TrafficLightEnv(max_queue=8, arrival_prob=0.5, max_steps=50)
    obs, info = env.reset(seed=42)
    print(f"✅ Environment created - Initial state: {obs}")
    
    # Test random actions
    print("\n3. Testing random actions...")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"   Step {step+1}: Action={action}, Reward={reward:.2f}, "
              f"NS={obs['ns_queue']}, EW={obs['ew_queue']}")
        if terminated or truncated:
            break
    print(f"✅ Random policy total reward: {total_reward:.2f}")
    
    # Create and train agent
    print("\n4. Training PPO agent...")
    agent = TrafficLightAgent(
        algorithm="PPO",
        env_config=config.get_environment_config(),
        model_config=config.get_model_config("PPO")
    )
    
    # Train with reduced timesteps for demo
    agent.train(total_timesteps=5000, log_interval=5)
    print("✅ PPO agent trained successfully")
    
    # Evaluate agent
    print("\n5. Evaluating trained agent...")
    mean_reward, std_reward = agent.evaluate(n_eval_episodes=5)
    print(f"✅ Evaluation complete - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Demo agent behavior
    print("\n6. Demonstrating agent behavior...")
    test_env = TrafficLightEnv(max_queue=8, arrival_prob=0.5, max_steps=20)
    obs, _ = test_env.reset(seed=123)
    
    agent_reward = 0
    for step in range(10):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        agent_reward += reward
        
        action_text = "Keep Light" if action == 0 else "Switch Light"
        light_text = "NS Green" if obs['light_state'] == 0 else "EW Green"
        
        print(f"   Step {step+1}: {action_text} → {light_text}, "
              f"NS={obs['ns_queue']}, EW={obs['ew_queue']}, Reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"✅ Agent total reward: {agent_reward:.2f}")
    
    # Performance comparison
    print("\n7. Performance Summary:")
    print(f"   Random Policy: {total_reward:.2f}")
    print(f"   Trained Agent: {agent_reward:.2f}")
    improvement = ((agent_reward - total_reward) / abs(total_reward)) * 100
    print(f"   Improvement: {improvement:.1f}%")
    
    # Create visualization
    print("\n8. Creating visualization...")
    visualizer = RLVisualizer(save_dir="demo_plots")
    
    # Generate dummy training data for visualization
    dummy_rewards = np.random.normal(mean_reward, std_reward, 100)
    visualizer.plot_training_curves(
        dummy_rewards, 
        algorithm="PPO Demo", 
        save=True, 
        show=False
    )
    print("✅ Visualization saved to demo_plots/")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("• Run 'python train.py --help' for more training options")
    print("• Run 'streamlit run app.py' for the web interface")
    print("• Check out notebooks/traffic_rl_tutorial.ipynb for interactive exploration")


if __name__ == "__main__":
    main()
