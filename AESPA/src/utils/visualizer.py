"""
Visualization utilities for prediction results
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_predictions_vs_targets(pred, target, save_path=None):
    """
    Plot predictions vs targets scatter plot
    
    Args:
        pred: [N] - predictions
        target: [N] - targets
        save_path: path to save figure
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(target, pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(target.min(), pred.min())
    max_val = max(target.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    
    plt.xlabel('Target Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title('Predictions vs Targets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_residuals(pred, target, save_path=None):
    """
    Plot residual plot
    
    Args:
        pred: [N] - predictions
        target: [N] - targets
        save_path: path to save figure
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    residuals = pred - target
    
    plt.figure(figsize=(8, 6))
    plt.scatter(target, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Target Temperature (°C)')
    plt.ylabel('Residuals (°C)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

