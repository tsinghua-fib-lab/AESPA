#!/usr/bin/env python
"""
Main entry point for AESPA training
"""
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import AESPADataset
from .models import TeacherModel, StudentModel
from .trainer import Trainer, StudentTrainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_dataloader(data_dir, split, config, batch_size, shuffle=False):
    """Create DataLoader for given split"""
    dataset = AESPADataset(data_dir, split=split, config=config['data'])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )


def train_teacher(config_path, data_dir, checkpoint_dir, log_dir, device, epochs, batch_size,
                  lambda_phys=0.05, lambda_proxy=0.0, lambda_ranking=0.1):
    """Train teacher model with physics-aware losses"""
    print("=" * 60)
    print("Training Teacher Model (Stage 1)")
    print("=" * 60)
    print(f"Loss weights: λ_phys={lambda_phys}, λ_proxy={lambda_proxy}, λ_ranking={lambda_ranking}")
    
    config = load_config(config_path)
    config['data']['data_dir'] = data_dir
    
    # Update loss weights from command line
    if 'losses' not in config:
        config['losses'] = {}
    config['losses']['alpha_physics'] = lambda_phys
    config['losses']['alpha_proxy'] = lambda_proxy
    config['losses']['alpha_ranking'] = lambda_ranking
    
    # Create model
    model = TeacherModel(config)
    
    # Create trainer
    trainer = Trainer(model, config, device=device)
    
    # Data loaders
    train_loader = get_dataloader(data_dir, 'train', config, batch_size, shuffle=True)
    val_loader = get_dataloader(data_dir, 'val', config, batch_size, shuffle=False)
    
    # TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    
    best_val_mae = float('inf')
    
    for epoch in range(epochs):
        trainer.epoch = epoch
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, writer=writer)
        print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader, writer=writer)
        print(f"Epoch {epoch}: Val MAE = {val_metrics['mae']:.4f}, "
              f"RMSE = {val_metrics['rmse']:.4f}, "
              f"R² = {val_metrics['r2']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['mae'] < best_val_mae
        if is_best:
            best_val_mae = val_metrics['mae']
        trainer.save_checkpoint(checkpoint_dir, is_best=is_best)
    
    writer.close()
    print(f"✅ Training completed. Best Val MAE: {best_val_mae:.4f}")


def train_student(config_path, data_dir, teacher_checkpoint, checkpoint_dir, log_dir, 
                  device, epochs, batch_size, lambda_kd=0.1, lambda_fd=0.05,
                  lambda_phys=0.05, lambda_ranking=0.1):
    """Train student model with distillation (Stage 2: Open Web deployment)"""
    print("=" * 60)
    print("Training Student Model (Stage 2: Knowledge Distillation)")
    print("=" * 60)
    print(f"Distillation weights: λ_kd={lambda_kd}, λ_fd={lambda_fd}")
    print(f"Loss weights: λ_phys={lambda_phys}, λ_ranking={lambda_ranking}")
    
    config = load_config(config_path)
    config['data']['data_dir'] = data_dir
    
    # Update loss weights from command line
    if 'losses' not in config:
        config['losses'] = {}
    config['losses']['alpha_physics'] = lambda_phys
    config['losses']['alpha_ranking'] = lambda_ranking
    
    if 'distillation' not in config:
        config['distillation'] = {}
    config['distillation']['alpha_kd'] = lambda_kd
    config['distillation']['alpha_fd'] = lambda_fd
    
    # Create models
    teacher_model = TeacherModel(config)
    student_model = StudentModel(config)
    
    # Load teacher checkpoint
    teacher_checkpoint_path = Path(teacher_checkpoint)
    if teacher_checkpoint_path.exists():
        checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded teacher checkpoint from {teacher_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")
    
    # Create trainer
    trainer = StudentTrainer(student_model, teacher_model, config, device=device)
    
    # Data loaders
    train_loader = get_dataloader(data_dir, 'train', config, batch_size, shuffle=True)
    val_loader = get_dataloader(data_dir, 'val', config, batch_size, shuffle=False)
    
    # TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    
    best_val_mae = float('inf')
    
    for epoch in range(epochs):
        trainer.epoch = epoch
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, writer=writer)
        print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader, writer=writer)
        print(f"Epoch {epoch}: Val MAE = {val_metrics['mae']:.4f}, "
              f"RMSE = {val_metrics['rmse']:.4f}, "
              f"R² = {val_metrics['r2']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['mae'] < best_val_mae
        if is_best:
            best_val_mae = val_metrics['mae']
        trainer.save_checkpoint(checkpoint_dir, is_best=is_best)
    
    writer.close()
    print(f"✅ Training completed. Best Val MAE: {best_val_mae:.4f}")


def main():
    parser = argparse.ArgumentParser(description='AESPA Training')
    parser.add_argument('--mode', type=str, required=True, choices=['teacher', 'student'],
                        help='Training mode: teacher or student')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--teacher-checkpoint', type=str, default=None,
                        help='Path to teacher checkpoint (required for student mode)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    # Loss weights for teacher training
    parser.add_argument('--lambda-phys', type=float, default=0.05,
                        help='Weight for physics consistency loss')
    parser.add_argument('--lambda-proxy', type=float, default=0.0,
                        help='Weight for proxy prediction loss')
    parser.add_argument('--lambda-ranking', type=float, default=0.1,
                        help='Weight for ranking loss (day > night)')
    # Distillation weights for student training
    parser.add_argument('--lambda-kd', type=float, default=0.1,
                        help='Weight for knowledge distillation loss')
    parser.add_argument('--lambda-fd', type=float, default=0.05,
                        help='Weight for feature distillation loss')
    
    args = parser.parse_args()
    
    if args.mode == 'teacher':
        train_teacher(
            args.config, args.data_dir, args.checkpoint_dir, args.log_dir,
            args.device, args.epochs, args.batch_size,
            lambda_phys=args.lambda_phys,
            lambda_proxy=args.lambda_proxy,
            lambda_ranking=args.lambda_ranking
        )
    elif args.mode == 'student':
        if args.teacher_checkpoint is None:
            raise ValueError("--teacher-checkpoint is required for student mode")
        train_student(
            args.config, args.data_dir, args.teacher_checkpoint,
            args.checkpoint_dir, args.log_dir, args.device,
            args.epochs, args.batch_size,
            lambda_kd=args.lambda_kd,
            lambda_fd=args.lambda_fd,
            lambda_phys=args.lambda_phys,
            lambda_ranking=args.lambda_ranking
        )


if __name__ == '__main__':
    main()

