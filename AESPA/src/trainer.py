"""
Training loop for Teacher and Student models

Implements Section 4.1: Two-stage training strategy
- Stage 1 (Teacher): Minimize L_reg + λ_phys * L_phys + λ_proxy * L_proxy + λ_rank * L_rank
- Stage 2 (Student): Freeze Teacher, minimize L_reg + λ_kd * L_kd + λ_fd * L_fd + λ_phys * L_phys + λ_rank * L_rank
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models import TeacherModel, StudentModel
from .models.losses import DistillationLoss, PhysicsConsistencyLoss, RankingLoss
from .utils.metrics import compute_all_metrics


class Trainer:
    """Trainer for Teacher or Student model"""
    
    def __init__(self, model, config, device='cuda:0'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.epoch = 0
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training'].get('lr', 1e-4),
            weight_decay=config['training'].get('weight_decay', 1e-5)
        )
        
        # Loss functions
        self.prediction_loss = nn.L1Loss()  # Regression loss for LST prediction
        self.physics_loss = PhysicsConsistencyLoss(config.get('losses', {}))  # L_phys
        self.ranking_loss = RankingLoss(margin=0.0)  # L_rank
        self.proxy_loss = nn.MSELoss()  # Proxy prediction loss (optional)
        
        # Loss weights (as per Section 4.1)
        self.alpha_pred = config['losses'].get('alpha_pred', 1.0)
        self.alpha_physics = config['losses'].get('alpha_physics', 0.05)  # λ_phys
        self.alpha_proxy = config['losses'].get('alpha_proxy', 0.0)  # λ_proxy (usually 0.0)
        self.alpha_ranking = config['losses'].get('alpha_ranking', 0.1)  # λ_ranking
        
    def train_epoch(self, train_loader, writer=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_physics_loss = 0.0
        total_proxy_loss = 0.0
        total_ranking_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            street_images = batch['street_images'].to(self.device)
            satellite_image = batch['satellite_image'].to(self.device)
            lst_labels = batch['lst_labels'].to(self.device)
            conditional_tokens = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                for k, v in batch['conditional_tokens'].items()
            }
            concept_targets = batch['concept_targets'].to(self.device)
            
            # Forward pass
            if isinstance(self.model, TeacherModel):
                mobility_data = batch.get('mobility_data')
                if mobility_data is not None:
                    mobility_data = mobility_data.to(self.device)
                lst_pred, concepts, features, night_pred = self.model(
                    street_images, satellite_image, mobility_data,
                    conditional_tokens, return_concepts=True
                )
            else:
                lst_pred, concepts, features, night_pred = self.model(
                    street_images, satellite_image,
                    conditional_tokens, return_concepts=True
                )
            
            # Prediction loss (day temperature)
            if lst_pred.dim() == 2 and lst_pred.shape[1] == 1:
                lst_pred_day = lst_pred[:, 0]
            else:
                lst_pred_day = lst_pred.squeeze()
            
            if lst_labels.dim() == 2 and lst_labels.shape[1] >= 1:
                lst_target_day = lst_labels[:, 0]
            else:
                lst_target_day = lst_labels.squeeze()
            
            pred_loss = self.prediction_loss(lst_pred_day, lst_target_day)
            
            # Physics consistency loss (L_phys)
            physics_loss_val = self.physics_loss(lst_pred, concepts, torch.ones_like(lst_pred_day, dtype=torch.bool))
            
            # Proxy prediction loss (optional, usually λ_proxy=0.0)
            proxy_loss_val = torch.tensor(0.0, device=self.device)
            if concepts is not None and self.alpha_proxy > 0:
                # Compute proxy loss if concept targets are available
                concept_preds = torch.stack([concepts.get(name, torch.zeros_like(lst_pred_day)) 
                                           for name in self.model.concept_names], dim=1)
                if concept_preds.shape[1] == concept_targets.shape[1]:
                    proxy_loss_val = self.proxy_loss(concept_preds, concept_targets)
            
            # Ranking loss (L_rank): ensure night < day
            ranking_loss_val = torch.tensor(0.0, device=self.device)
            if night_pred is not None and lst_labels.shape[1] >= 2:
                lst_target_night = lst_labels[:, 1]
                ranking_loss_val = self.ranking_loss(lst_pred_day, night_pred.squeeze())
            
            # Total loss: L = L_reg + λ_phys * L_phys + λ_proxy * L_proxy + λ_rank * L_rank
            loss = (
                self.alpha_pred * pred_loss +
                self.alpha_physics * physics_loss_val +
                self.alpha_proxy * proxy_loss_val +
                self.alpha_ranking * ranking_loss_val
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_physics_loss += physics_loss_val.item()
            total_proxy_loss += proxy_loss_val.item()
            total_ranking_loss += ranking_loss_val.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pred': f'{pred_loss.item():.4f}',
            })
            
            # Log to tensorboard
            if writer and batch_idx % 100 == 0:
                global_step = self.epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/pred_loss', pred_loss.item(), global_step)
                writer.add_scalar('train/physics_loss', physics_loss_val.item(), global_step)
        
        avg_loss = total_loss / len(train_loader)
        avg_pred_loss = total_pred_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)
        avg_proxy_loss = total_proxy_loss / len(train_loader)
        avg_ranking_loss = total_ranking_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'pred_loss': avg_pred_loss,
            'physics_loss': avg_physics_loss,
            'proxy_loss': avg_proxy_loss,
            'ranking_loss': avg_ranking_loss,
        }
    
    def validate(self, val_loader, writer=None):
        """Validate model"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                street_images = batch['street_images'].to(self.device)
                satellite_image = batch['satellite_image'].to(self.device)
                lst_labels = batch['lst_labels'].to(self.device)
                conditional_tokens = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                    for k, v in batch['conditional_tokens'].items()
                }
                
                if isinstance(self.model, TeacherModel):
                    mobility_data = batch.get('mobility_data')
                    if mobility_data is not None:
                        mobility_data = mobility_data.to(self.device)
                    lst_pred, _, _, _ = self.model(
                        street_images, satellite_image, mobility_data,
                        conditional_tokens, return_concepts=False
                    )
                else:
                    lst_pred, _, _, _ = self.model(
                        street_images, satellite_image,
                        conditional_tokens, return_concepts=False
                    )
                
                if lst_pred.dim() == 2 and lst_pred.shape[1] == 1:
                    lst_pred_day = lst_pred[:, 0]
                else:
                    lst_pred_day = lst_pred.squeeze()
                
                if lst_labels.dim() == 2 and lst_labels.shape[1] >= 1:
                    lst_target_day = lst_labels[:, 0]
                else:
                    lst_target_day = lst_labels.squeeze()
                
                all_preds.append(lst_pred_day.cpu())
                all_targets.append(lst_target_day.cpu())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = compute_all_metrics(all_preds, all_targets)
        
        if writer:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(f'val/{metric_name}', metric_value, self.epoch)
        
        return metrics
    
    def save_checkpoint(self, checkpoint_dir, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']


class StudentTrainer(Trainer):
    """Trainer for Student model with distillation"""
    
    def __init__(self, student_model, teacher_model, config, device='cuda:0'):
        super().__init__(student_model, config, device)
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()  # Freeze teacher
        
        # Distillation loss
        self.distillation_loss = DistillationLoss(config.get('distillation', {}))
        self.alpha_distill = config['losses'].get('alpha_distill', 0.5)
    
    def train_epoch(self, train_loader, writer=None):
        """Train student with distillation"""
        self.model.train()
        self.teacher_model.eval()
        
        total_loss = 0.0
        total_pred_loss = 0.0
        total_distill_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            street_images = batch['street_images'].to(self.device)
            satellite_image = batch['satellite_image'].to(self.device)
            lst_labels = batch['lst_labels'].to(self.device)
            conditional_tokens = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                for k, v in batch['conditional_tokens'].items()
            }
            
            # Student forward
            lst_pred_student, _, features_student, _ = self.model(
                street_images, satellite_image,
                conditional_tokens, return_concepts=False
            )
            
            # Teacher forward (no grad)
            with torch.no_grad():
                mobility_data = batch.get('mobility_data')
                if mobility_data is not None:
                    mobility_data = mobility_data.to(self.device)
                lst_pred_teacher, _, features_teacher, _ = self.teacher_model(
                    street_images, satellite_image, mobility_data,
                    conditional_tokens, return_concepts=False
                )
            
            # Prediction loss
            if lst_pred_student.dim() == 2 and lst_pred_student.shape[1] == 1:
                lst_pred_day = lst_pred_student[:, 0]
            else:
                lst_pred_day = lst_pred_student.squeeze()
            
            if lst_labels.dim() == 2 and lst_labels.shape[1] >= 1:
                lst_target_day = lst_labels[:, 0]
            else:
                lst_target_day = lst_labels.squeeze()
            
            pred_loss = self.prediction_loss(lst_pred_day, lst_target_day)
            
            # Distillation loss
            if lst_pred_teacher.dim() == 2 and lst_pred_teacher.shape[1] == 1:
                lst_pred_teacher_day = lst_pred_teacher[:, 0]
            else:
                lst_pred_teacher_day = lst_pred_teacher.squeeze()
            
            distill_loss, distill_losses = self.distillation_loss(
                lst_pred_day, lst_pred_teacher_day,
                features_student, features_teacher
            )
            
            # Total loss
            loss = self.alpha_pred * pred_loss + self.alpha_distill * distill_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_distill_loss += distill_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pred': f'{pred_loss.item():.4f}',
                'distill': f'{distill_loss.item():.4f}',
            })
            
            if writer and batch_idx % 100 == 0:
                global_step = self.epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/pred_loss', pred_loss.item(), global_step)
                writer.add_scalar('train/distill_loss', distill_loss.item(), global_step)
        
        return {
            'loss': total_loss / len(train_loader),
            'pred_loss': total_pred_loss / len(train_loader),
            'distill_loss': total_distill_loss / len(train_loader),
        }

