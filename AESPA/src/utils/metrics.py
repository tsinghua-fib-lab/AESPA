"""
Metrics for evaluation: MAE, RMSE, R², Pearson Correlation
"""
import numpy as np
import torch


def compute_mae(pred, target):
    """Mean Absolute Error"""
    return float(torch.mean(torch.abs(pred - target)).item())


def compute_rmse(pred, target):
    """Root Mean Square Error"""
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def compute_r2(pred, target):
    """R² Score"""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2) + 1e-8
    return float(1.0 - ss_res / ss_tot)


def compute_pearson(pred, target):
    """Pearson Correlation Coefficient"""
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    numerator = (pred_centered * target_centered).mean()
    denominator = (pred_centered.std() * target_centered.std()) + 1e-8
    return float((numerator / denominator).item())


def compute_all_metrics(pred, target):
    """Compute all metrics"""
    return {
        'mae': compute_mae(pred, target),
        'rmse': compute_rmse(pred, target),
        'r2': compute_r2(pred, target),
        'pearson': compute_pearson(pred, target),
    }

