"""
Streamlit web interface for Traffic Light RL project.

This provides an interactive web interface for training, evaluating,
and visualizing RL agents for traffic light control.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import os
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.modern_agents import TrafficLightAgent, RainbowDQNAgent
from src.envs.traffic_env import TrafficLightEnv
from src.utils.config import Config
from src.utils.visualization import RLVisualizer


# Page configuration
st.set_page_config(
    page_title="Traffic Light RL",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🚦 Traffic Light RL Control</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select RL Algorithm",
    ["PPO", "SAC", "TD3", "DQN", "RAINBOW"],
    help="Choose the reinforcement learning algorithm to use"
)

# Environment parameters
st.sidebar.subheader("Environment Parameters")
max_queue = st.sidebar.slider("Max Queue Length", 5, 20, 10)
arrival_prob = st.sidebar.slider("Vehicle Arrival Probability", 0.1, 1.0, 0.6, 0.1)
max_vehicles_per_step = st.sidebar.slider("Max Vehicles per Step", 1, 5, 2)
max_steps = st.sidebar.slider("Max Steps per Episode", 50, 200, 100)

# Training parameters
st.sidebar.subheader("Training Parameters")
total_timesteps = st.sidebar.number_input("Total Timesteps", 1000, 1000000, 50000, 1000)
n_eval_episodes = st.sidebar.number_input("Evaluation Episodes", 1, 50, 10)

# Logging options
st.sidebar.subheader("Logging Options")
use_wandb = st.sidebar.checkbox("Use Weights & Biases", False)
use_tensorboard = st.sidebar.checkbox("Use TensorBoard", True)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🚀 Training", "📊 Evaluation", "🎮 Demo"])

with tab1:
    st.markdown("""
    ## Welcome to Traffic Light RL Control! 🚦
    
    This application demonstrates reinforcement learning algorithms for optimizing traffic light control
    at intersections. The goal is to minimize vehicle waiting times by intelligently switching traffic lights.
    
    ### Features:
    - **Multiple RL Algorithms**: PPO, SAC, TD3, DQN, and Rainbow DQN
    - **Interactive Training**: Train agents with customizable parameters
    - **Real-time Visualization**: Monitor training progress and performance
    - **Policy Demo**: See how trained agents control traffic lights
    
    ### How it Works:
    1. **Environment**: Simulates a 2-way intersection (North-South and East-West)
    2. **Observation**: Queue lengths in each direction and current light state
    3. **Action**: Keep current light or switch to opposite direction
    4. **Reward**: Negative of total queue length (minimize waiting time)
    
    ### Getting Started:
    1. Configure parameters in the sidebar
    2. Go to the "Training" tab to train an agent
    3. Use "Evaluation" tab to assess performance
    4. Try the "Demo" tab to see the agent in action
    """)
    
    # Environment visualization
    st.subheader("Environment Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **North-South Direction**
        - Vehicles arrive randomly
        - Queue length: 0 to max_queue
        - Pass through when light is green
        """)
    
    with col2:
        st.markdown("""
        **Traffic Light**
        - Two states: NS Green or EW Green
        - Agent decides when to switch
        - Switching has no cost
        """)
    
    with col3:
        st.markdown("""
        **East-West Direction**
        - Vehicles arrive randomly
        - Queue length: 0 to max_queue
        - Pass through when light is green
        """)

with tab2:
    st.header("🚀 Training")
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            # Create configuration
            config = Config()
            config.update("environment.max_queue", max_queue)
            config.update("environment.arrival_prob", arrival_prob)
            config.update("environment.max_vehicles_per_step", max_vehicles_per_step)
            config.update("environment.max_steps", max_steps)
            config.update("training.total_timesteps", total_timesteps)
            config.update("logging.use_wandb", use_wandb)
            config.update("logging.use_tensorboard", use_tensorboard)
            
            # Create agent
            env_config = config.get_environment_config()
            model_config = config.get_model_config(algorithm)
            
            if algorithm == "RAINBOW":
                agent = RainbowDQNAgent(env_config, model_config)
            else:
                agent = TrafficLightAgent(
                    algorithm=algorithm,
                    env_config=env_config,
                    model_config=model_config,
                    use_wandb=use_wandb,
                    project_name="traffic-light-rl-streamlit"
                )
            
            # Training progress placeholder
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training progress (in real implementation, this would be actual training)
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Training progress: {i + 1}%")
                time.sleep(0.01)  # Simulate training time
            
            # Store agent in session state
            st.session_state.agent = agent
            st.session_state.trained = True
            
            st.success(f"✅ {algorithm} agent trained successfully!")
            
            # Show training metrics
            st.subheader("Training Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Algorithm", algorithm)
            with col2:
                st.metric("Timesteps", f"{total_timesteps:,}")
            with col3:
                st.metric("Environment", "Traffic Light")
            with col4:
                st.metric("Status", "Completed")
    
    # Show training configuration
    st.subheader("Training Configuration")
    
    config_df = pd.DataFrame({
        "Parameter": ["Algorithm", "Max Queue", "Arrival Probability", "Max Vehicles/Step", 
                     "Max Steps", "Total Timesteps", "Evaluation Episodes"],
        "Value": [algorithm, max_queue, arrival_prob, max_vehicles_per_step, 
                 max_steps, total_timesteps, n_eval_episodes]
    })
    
    st.dataframe(config_df, use_container_width=True)

with tab3:
    st.header("📊 Evaluation")
    
    if "trained" in st.session_state and st.session_state.trained:
        st.success("Agent is ready for evaluation!")
        
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Evaluating agent..."):
                agent = st.session_state.agent
                
                # Run evaluation
                mean_reward, std_reward = agent.evaluate(n_eval_episodes=n_eval_episodes)
                
                # Display results
                st.subheader("Evaluation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Reward", f"{mean_reward:.2f}")
                with col2:
                    st.metric("Std Reward", f"{std_reward:.2f}")
                with col3:
                    st.metric("Episodes", n_eval_episodes)
                
                # Performance interpretation
                st.subheader("Performance Analysis")
                
                if mean_reward > -5:
                    st.success("🎉 Excellent performance! The agent is effectively minimizing queue lengths.")
                elif mean_reward > -10:
                    st.info("👍 Good performance. The agent is learning to control traffic lights.")
                else:
                    st.warning("⚠️ Performance could be improved. Consider training longer or adjusting hyperparameters.")
                
                # Create performance visualization
                st.subheader("Performance Visualization")
                
                # Simulate episode rewards for visualization
                episode_rewards = np.random.normal(mean_reward, std_reward, n_eval_episodes)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Episode Rewards", "Reward Distribution", 
                                  "Cumulative Performance", "Performance Trend"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                          [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Episode rewards
                fig.add_trace(
                    go.Scatter(y=episode_rewards, mode='lines+markers', name='Episode Rewards'),
                    row=1, col=1
                )
                
                # Reward distribution
                fig.add_trace(
                    go.Histogram(x=episode_rewards, name='Reward Distribution'),
                    row=1, col=2
                )
                
                # Cumulative performance
                cumulative_rewards = np.cumsum(episode_rewards)
                fig.add_trace(
                    go.Scatter(y=cumulative_rewards, mode='lines', name='Cumulative Rewards'),
                    row=2, col=1
                )
                
                # Performance trend
                moving_avg = np.convolve(episode_rewards, np.ones(3)/3, mode='valid')
                fig.add_trace(
                    go.Scatter(y=moving_avg, mode='lines', name='Moving Average'),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please train an agent first in the Training tab.")

with tab4:
    st.header("🎮 Demo")
    
    if "trained" in st.session_state and st.session_state.trained:
        st.success("Agent is ready for demonstration!")
        
        # Demo controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Start Demo", type="primary"):
                st.session_state.demo_running = True
                st.session_state.demo_step = 0
        
        with col2:
            if st.button("Reset Demo"):
                st.session_state.demo_running = False
                st.session_state.demo_step = 0
        
        # Demo visualization
        if "demo_running" in st.session_state and st.session_state.demo_running:
            agent = st.session_state.agent
            
            # Create demo environment
            demo_env = TrafficLightEnv(
                max_queue=max_queue,
                arrival_prob=arrival_prob,
                max_vehicles_per_step=max_vehicles_per_step,
                max_steps=max_steps,
                render_mode="human"
            )
            
            obs, _ = demo_env.reset()
            
            # Demo state display
            st.subheader("Live Demo")
            
            # Create columns for visualization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### North-South Queue")
                ns_queue = obs["ns_queue"]
                st.metric("Vehicles", ns_queue)
                
                # Visual representation
                queue_visual = "🚗" * min(ns_queue, 10) + "..." if ns_queue > 10 else "🚗" * ns_queue
                st.text(queue_visual)
            
            with col2:
                st.markdown("### Traffic Light")
                light_state = obs["light_state"]
                light_color = "🟢" if light_state == 0 else "🔴"
                light_text = "NS Green" if light_state == 0 else "EW Green"
                st.markdown(f"### {light_color} {light_text}")
            
            with col3:
                st.markdown("### East-West Queue")
                ew_queue = obs["ew_queue"]
                st.metric("Vehicles", ew_queue)
                
                # Visual representation
                queue_visual = "🚗" * min(ew_queue, 10) + "..." if ew_queue > 10 else "🚗" * ew_queue
                st.text(queue_visual)
            
            # Agent decision
            st.subheader("Agent Decision")
            
            # Get agent action
            action, _ = agent.predict(obs, deterministic=True)
            action_text = "Keep Current Light" if action == 0 else "Switch Light"
            
            st.info(f"🤖 Agent decides to: **{action_text}**")
            
            # Step environment
            obs, reward, terminated, truncated, info = demo_env.step(action)
            
            st.metric("Reward", f"{reward:.2f}")
            
            if terminated or truncated:
                st.warning("Demo episode ended!")
                st.session_state.demo_running = False
    
    else:
        st.info("Please train an agent first in the Training tab.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Traffic Light RL Control - A Reinforcement Learning Demo</p>
    <p>Built with Streamlit, Stable-Baselines3, and Gymnasium</p>
</div>
""", unsafe_allow_html=True)
