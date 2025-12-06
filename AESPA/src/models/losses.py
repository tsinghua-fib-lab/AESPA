"""
Loss Functions for AESPA: Physics Consistency, Ranking, Distillation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Distillation Loss combining feature distillation and knowledge distillation"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config.get('temperature', 2.0)
        self.alpha_fd = config.get('alpha_fd', 0.3)  # Feature distillation weight
        self.alpha_kd = config.get('alpha_kd', 0.5)  # Knowledge distillation weight
    
    def feature_distillation_loss(self, student_features, teacher_features):
        """
        Feature distillation: L_fd = ||z_student - stopgrad(z_teacher)||_2^2 / D
        Normalized by feature dimension to match the scale of L1 loss.
        """
        teacher_features = teacher_features.detach()  # stopgrad
        mse_loss = F.mse_loss(student_features, teacher_features)
        # Normalize by feature dimension to match the scale of prediction loss
        feature_dim = student_features.shape[-1]
        return mse_loss / feature_dim
    
    def knowledge_distillation_loss(self, student_pred, teacher_pred):
        """
        Knowledge distillation for regression (Section 3.8): 
        L_kd = |ŷ_student - ŷ_teacher| (L1 loss)
        """
        teacher_pred = teacher_pred.detach()  # stopgrad
        
        # Ensure same shape
        if student_pred.dim() == 1:
            student_pred = student_pred.unsqueeze(-1)
        if teacher_pred.dim() == 1:
            teacher_pred = teacher_pred.unsqueeze(-1)
        
        # Use L1 loss as per paper Section 3.8
        return F.l1_loss(student_pred, teacher_pred)
    
    def forward(self, student_lst, teacher_lst, student_features, teacher_features):
        """
        Args:
            student_lst: [B, 1] or [B] - student LST predictions (day only)
            teacher_lst: [B, 1] or [B] - teacher LST predictions (day only)
            student_features: [B, D] - student features
            teacher_features: [B, D] - teacher features
        Returns:
            total_loss: scalar - total distillation loss
            losses: dict - individual loss components
        """
        # Feature distillation
        fd_loss = self.feature_distillation_loss(student_features, teacher_features)
        
        # Knowledge distillation
        if student_lst.dim() == 1:
            student_lst = student_lst.unsqueeze(-1)
        if teacher_lst.dim() == 1:
            teacher_lst = teacher_lst.unsqueeze(-1)
        kd_loss = self.knowledge_distillation_loss(student_lst, teacher_lst)
        
        # Total loss
        total_loss = self.alpha_fd * fd_loss + self.alpha_kd * kd_loss
        
        losses = {
            'feature_distillation': fd_loss,
            'knowledge_distillation': kd_loss,
            'total_distillation': total_loss
        }
        
        return total_loss, losses


class PhysicsConsistencyLoss(nn.Module):
    """
    Physics Consistency Loss: L_phys = Σ max(0, -s * correlation)
    
    Implements physics constraints via Pearson correlation with sign constraints:
    - NDVI ↑ → LST ↓ (s = -1): max(0, -(-1) * corr) = max(0, corr)
    - Impervious ↑ → LST ↑ (s = +1): max(0, -(+1) * corr) = max(0, -corr)
    - Albedo ↑ → Day LST ↓ (s = -1): max(0, corr)
    - Shadow ↑ → LST ↓ (s = -1): max(0, corr)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.day_only = config.get('day_only', True)  # Apply only to day predictions
    
    def forward(self, lst_pred, concepts, is_day):
        """
        Physics constraints with sign-based correlation:
        - NDVI ↑ → LST ↓: enforce negative correlation (s = -1)
        - Impervious ↑ → LST ↑: enforce positive correlation (s = +1)
        - High albedo → Day LST ↓: enforce negative correlation (s = -1)
        - Shadow ↑ → LST ↓: enforce negative correlation (s = -1)
        
        Loss = max(0, -s * correlation) where s is the expected sign
        
        Args:
            lst_pred: [B, 1] or [B] - predicted LST (day only)
            concepts: dict - concept predictions
            is_day: [B] - boolean mask for day samples
        Returns:
            loss: scalar - physics consistency loss
        """
        batch_size = lst_pred.shape[0]
        if batch_size < 2:
            return lst_pred.new_tensor(0.0)

        loss = lst_pred.new_tensor(0.0)
        # Handle both [B, 1] and [B] shapes
        if lst_pred.dim() == 2 and lst_pred.shape[1] == 1:
            lst_day = lst_pred[:, 0]
        elif lst_pred.dim() == 1:
            lst_day = lst_pred
        else:
            lst_day = lst_pred[:, 0]  # Fallback to first column

        def safe_correlation(x, y):
            """Compute Pearson correlation coefficient"""
            x = x - x.mean()
            y = y - y.mean()
            denom = (x.std() * y.std()) + 1e-6
            return (x * y).mean() / denom

        # NDVI ↑ → LST ↓ (expected sign s = -1)
        # Loss = max(0, -(-1) * corr) = max(0, corr)
        if 'ndvi_proxy' in concepts:
            ndvi = concepts['ndvi_proxy']
            corr = safe_correlation(ndvi, lst_day)
            s = -1  # Expected negative correlation
            loss = loss + F.relu(-s * corr)  # max(0, corr)

        # Impervious ↑ → LST ↑ (expected sign s = +1)
        # Loss = max(0, -(+1) * corr) = max(0, -corr)
        if 'impervious_proxy' in concepts:
            impervious = concepts['impervious_proxy']
            corr = safe_correlation(impervious, lst_day)
            s = +1  # Expected positive correlation
            loss = loss + F.relu(-s * corr)  # max(0, -corr)

        # Albedo ↑ → Day LST ↓ (expected sign s = -1)
        # Loss = max(0, corr)
        if 'albedo_proxy' in concepts and self.day_only:
            albedo = concepts['albedo_proxy']
            corr = safe_correlation(albedo, lst_day)
            s = -1  # Expected negative correlation
            loss = loss + F.relu(-s * corr)  # max(0, corr)

        # Shadow ↑ → LST ↓ (expected sign s = -1)
        # Loss = max(0, corr)
        if 'shadow_fraction' in concepts:
            shadow = concepts['shadow_fraction']
            corr = safe_correlation(shadow, lst_day)
            s = -1  # Expected negative correlation
            loss = loss + F.relu(-s * corr)  # max(0, corr)

        return loss


class RankingLoss(nn.Module):
    """
    Ranking Loss for day/night temperature constraint.
    
    Ensures that night temperature < day temperature (physical constraint).
    Loss = max(0, night_temp - day_temp + margin)
    """
    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, day_pred, night_pred):
        """
        Args:
            day_pred: [B, 1] or [B] - predicted day temperature
            night_pred: [B, 1] or [B] - predicted night temperature
        Returns:
            loss: scalar - ranking loss
        """
        # Ensure both are [B] shape
        if day_pred.dim() == 2:
            day_pred = day_pred.squeeze(-1)
        if night_pred.dim() == 2:
            night_pred = night_pred.squeeze(-1)
        
        # Ranking constraint: night < day
        # Loss = max(0, night - day + margin)
        violation = night_pred - day_pred + self.margin
        loss = torch.clamp(violation, min=0.0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

