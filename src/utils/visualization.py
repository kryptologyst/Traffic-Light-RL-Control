"""
Visualization utilities for the Traffic Light RL project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path


class RLVisualizer:
    """Visualization utilities for RL training and evaluation."""
    
    def __init__(self, save_dir: str = "plots", style: str = "seaborn-v0_8"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.save_dir = save_dir
        self.style = style
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_training_curves(self, 
                           rewards: List[float],
                           algorithm: str = "RL Agent",
                           window_size: int = 100,
                           save: bool = True,
                           show: bool = True) -> None:
        """
        Plot training reward curves.
        
        Args:
            rewards: List of episode rewards
            algorithm: Name of the algorithm
            window_size: Window size for moving average
            save: Whether to save the plot
            show: Whether to show the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Raw rewards
        ax1.plot(rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
        
        # Moving average
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(rewards)), moving_avg, 
                    color='darkblue', linewidth=2, label=f'Moving Average ({window_size})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title(f'{algorithm} - Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Reward distribution
        ax2.hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(rewards):.2f}')
        ax2.axvline(np.median(rewards), color='green', linestyle='--', 
                   label=f'Median: {np.median(rewards):.2f}')
        ax2.set_xlabel('Episode Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{algorithm.lower().replace(' ', '_')}_training_curves.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_algorithm_comparison(self, 
                                results: Dict[str, List[float]],
                                window_size: int = 100,
                                save: bool = True,
                                show: bool = True) -> None:
        """
        Compare multiple algorithms.
        
        Args:
            results: Dictionary mapping algorithm names to reward lists
            window_size: Window size for moving average
            save: Whether to save the plot
            show: Whether to show the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for i, (algorithm, rewards) in enumerate(results.items()):
            # Plot raw rewards with transparency
            ax.plot(rewards, alpha=0.2, color=colors[i])
            
            # Plot moving average
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax.plot(range(window_size-1, len(rewards)), moving_avg, 
                       color=colors[i], linewidth=2, label=algorithm)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Algorithm Comparison - Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = "algorithm_comparison.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_evaluation_results(self, 
                              results: Dict[str, Dict[str, float]],
                              save: bool = True,
                              show: bool = True) -> None:
        """
        Plot evaluation results comparison.
        
        Args:
            results: Dictionary mapping algorithm names to evaluation metrics
            save: Whether to save the plot
            show: Whether to show the plot
        """
        algorithms = list(results.keys())
        metrics = list(results[algorithms[0]].keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [results[alg][metric] for alg in algorithms]
            
            bars = axes[i].bar(algorithms, values, color=plt.cm.Set1(np.linspace(0, 1, len(algorithms))))
            axes[i].set_title(f'{metric}')
            axes[i].set_ylabel(metric)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        plt.suptitle('Algorithm Evaluation Results', fontsize=16)
        plt.tight_layout()
        
        if save:
            filename = "evaluation_results.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Evaluation results saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_policy_heatmap(self, 
                          policy_data: np.ndarray,
                          title: str = "Policy Heatmap",
                          save: bool = True,
                          show: bool = True) -> None:
        """
        Plot policy as a heatmap.
        
        Args:
            policy_data: 2D array representing policy values
            title: Title for the plot
            save: Whether to save the plot
            show: Whether to show the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(policy_data, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Action Value')
        
        ax.set_title(title)
        ax.set_xlabel('East-West Queue Length')
        ax.set_ylabel('North-South Queue Length')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{title.lower().replace(' ', '_')}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Policy heatmap saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_state_value_function(self, 
                                value_function: np.ndarray,
                                title: str = "State Value Function",
                                save: bool = True,
                                show: bool = True) -> None:
        """
        Plot state value function as a heatmap.
        
        Args:
            value_function: 2D array representing state values
            title: Title for the plot
            save: Whether to save the plot
            show: Whether to show the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(value_function, cmap='plasma', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('State Value')
        
        ax.set_title(title)
        ax.set_xlabel('East-West Queue Length')
        ax.set_ylabel('North-South Queue Length')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{title.lower().replace(' ', '_')}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"State value function saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_summary_report(self, 
                            training_results: Dict[str, List[float]],
                            evaluation_results: Dict[str, Dict[str, float]],
                            save: bool = True) -> None:
        """
        Create a comprehensive summary report.
        
        Args:
            training_results: Training results for each algorithm
            evaluation_results: Evaluation results for each algorithm
            save: Whether to save the report
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Training curves
        ax1 = fig.add_subplot(gs[0, :])
        colors = plt.cm.Set1(np.linspace(0, 1, len(training_results)))
        
        for i, (algorithm, rewards) in enumerate(training_results.items()):
            window_size = 100
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(rewards)), moving_avg, 
                        color=colors[i], linewidth=2, label=algorithm)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training Progress Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Evaluation metrics
        if evaluation_results:
            algorithms = list(evaluation_results.keys())
            metrics = list(evaluation_results[algorithms[0]].keys())
            
            for i, metric in enumerate(metrics):
                ax = fig.add_subplot(gs[1, i])
                values = [evaluation_results[alg][metric] for alg in algorithms]
                
                bars = ax.bar(algorithms, values, color=colors[:len(algorithms)])
                ax.set_title(f'{metric}')
                ax.set_ylabel(metric)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Statistics table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        
        # Create statistics table
        stats_data = []
        for algorithm, rewards in training_results.items():
            stats_data.append([
                algorithm,
                f"{np.mean(rewards):.2f}",
                f"{np.std(rewards):.2f}",
                f"{np.min(rewards):.2f}",
                f"{np.max(rewards):.2f}",
                f"{np.median(rewards):.2f}"
            ])
        
        table = ax_table.table(cellText=stats_data,
                             colLabels=['Algorithm', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                             cellLoc='center',
                             loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax_table.set_title('Training Statistics', pad=20)
        
        plt.suptitle('Traffic Light RL - Summary Report', fontsize=16, y=0.98)
        
        if save:
            filename = "summary_report.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Summary report saved to: {filepath}")
        
        plt.show()
