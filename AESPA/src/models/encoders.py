"""
Encoders for AESPA (Section 3.1):
- Satellite Encoder: ViT-Base feature extractor with Adapter layers
- Street View Encoder: CLIP ViT-B/16 with LoRA + Attention-based MIL aggregation
- Mobility Encoder: GRU network for 24×7 (168-dim) mobility features
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, ViTModel
from peft import LoraConfig, get_peft_model, TaskType

# ==== 环境变量（确保使用本地镜像/缓存） ====
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
os.environ.setdefault("CURL_CA_BUNDLE", "")


class GatedAttentionMIL(nn.Module):
    """对实例特征执行门控注意力聚合"""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor):
        """features: [B, K, D]"""
        if features.dim() != 3:
            raise ValueError(f"GatedAttentionMIL 期望 3D 张量，收到 shape={features.shape}")

        V = self.attention_V(features)  # [B, K, H]
        U = self.attention_U(features)  # [B, K, H]
        logits = self.attention_weights(V * U)  # [B, K, 1]
        weights = torch.softmax(logits, dim=1)  # [B, K, 1]

        aggregated = torch.sum(features * weights, dim=1)  # [B, D]
        return aggregated, weights.squeeze(-1)


class StreetViewEncoder(nn.Module):
    """基于 CLIP ViT-B/16 的街景图像编码器，支持 LoRA 与 MIL 聚合"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_images = config.get("num_images", 16)
        self.input_size = config.get("input_size", 224)
        self.hidden_dim = config.get("hidden_dim", 512)

        model_path = Path(config.get("clip_local_dir", "/data6/youyuanyi/clip-vit-base-patch16"))
        if not model_path.exists():
            raise FileNotFoundError(f"未找到 CLIP 本地权重目录: {model_path}")

        # 直接从本地加载 CLIP 模型
        self.clip_model = CLIPModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        # 冻结骨干
        if config.get("freeze_backbone", True):
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # 应用 LoRA（可选）
        if config.get("use_lora", True):
            lora_task_type = getattr(TaskType, "VISION", TaskType.FEATURE_EXTRACTION)
            lora_cfg = LoraConfig(
                task_type=lora_task_type,
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.1),
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            self.clip_model.vision_model = get_peft_model(self.clip_model.vision_model, lora_cfg)

        self.vision_dim = self.clip_model.config.vision_config.hidden_size
        self.proj = nn.Linear(self.vision_dim, self.hidden_dim)

        mil_type = config.get("mil_type", "gated_attention")
        if mil_type == "gated_attention":
            self.mil = GatedAttentionMIL(self.hidden_dim, config.get("mil_hidden", 256))
        elif mil_type == "mean":
            self.mil = None  # 使用简单平均
        else:
            raise ValueError(f"不支持的 mil_type: {mil_type}")

    def forward(self, street_images: torch.Tensor):
        """
        Args:
            street_images: [B, K, C, H, W] 或 [B, C, H, W]
        Returns:
            features: [B, hidden_dim]
            attention_weights: [B, K] 或 None
        """
        if street_images.dim() == 4:
            street_images = street_images.unsqueeze(1)
        if street_images.dim() != 5:
            raise ValueError(f"StreetViewEncoder 期望 shape=[B,K,C,H,W]，收到 {street_images.shape}")

        B, K, C, H, W = street_images.shape
        street_images = street_images.view(B * K, C, H, W)

        vision_module = self.clip_model.vision_model
        if hasattr(vision_module, "base_model"):
            vision_outputs = vision_module.base_model(pixel_values=street_images)
        else:
            vision_outputs = vision_module(pixel_values=street_images)
        cls_features = vision_outputs.last_hidden_state[:, 0, :]  # [B*K, vision_dim]
        projected = self.proj(cls_features)  # [B*K, hidden_dim]
        projected = projected.view(B, K, self.hidden_dim)

        if self.mil is None:
            aggregated = projected.mean(dim=1)
            attention = None
        else:
            aggregated, attention = self.mil(projected)

        return aggregated, attention


class AdapterLayer(nn.Module):
    """Adapter layer for fine-tuning"""
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return x + self.up_proj(self.activation(self.down_proj(x)))


class SatelliteEncoder(nn.Module):
    """Satellite Encoder with ViT-Base + Adapter"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.get('input_size', 256)
        
        vit_local_dir = config.get('vit_local_dir')
        if vit_local_dir is None:
            raise ValueError(
                "SatelliteEncoder: 请在配置的 satellite 段中设置 vit_local_dir 指向本地 ViT 权重目录，以避免联网下载。"
            )
        vit_path = Path(vit_local_dir)
        if not vit_path.exists():
            raise FileNotFoundError(
                f"SatelliteEncoder: 未找到本地 ViT 权重目录 {vit_path}. "
                "请将 google/vit-base-patch16-224 下载到该路径（保持 transformers 格式），"
                "或在配置中修改 vit_local_dir 为实际目录。"
            )
        self.vit_model = ViTModel.from_pretrained(
            str(vit_path),
            local_files_only=True
        )
        self.hidden_dim = 768
        
        # Freeze backbone if specified
        if config.get('freeze_backbone', True):
            for param in self.vit_model.parameters():
                param.requires_grad = False
        
        # Add adapters to transformer blocks
        self.use_adapter = config.get('use_adapter', True)
        if self.use_adapter:
            bottleneck_dim = config.get('adapter_bottleneck', 64)
            for layer in self.vit_model.encoder.layer:
                layer.adapter = AdapterLayer(self.hidden_dim, bottleneck_dim)
        
        # Projection layer
        self.proj = nn.Linear(self.hidden_dim, config.get('hidden_dim', 512))
    
    def forward(self, satellite_image):
        """
        Args:
            satellite_image: [B, C, H, W] - satellite image
        Returns:
            features: [B, D] - satellite features
        """
        # Resize to model's expected input size (224x224 for ViT)
        model_input_size = 224
        if satellite_image.shape[-1] != model_input_size or satellite_image.shape[-2] != model_input_size:
            satellite_image = F.interpolate(
                satellite_image,
                size=(model_input_size, model_input_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Extract features
        outputs = self.vit_model(pixel_values=satellite_image)
        features = outputs.last_hidden_state
        
        # Use CLS token (first token)
        cls_features = features[:, 0, :]  # [B, hidden_dim]
        
        # Apply adapters if enabled
        if self.use_adapter:
            for layer in self.vit_model.encoder.layer:
                if hasattr(layer, 'adapter'):
                    cls_features = layer.adapter(cls_features)
        
        # Project to hidden dimension
        projected = self.proj(cls_features)  # [B, hidden_dim]
        
        return projected


class MobilityEncoder(nn.Module):
    """Mobility Encoder using GRU, MLP, or TCN for 24×7 (168-dim) mobility features"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_type = config.get('encoder_type', 'gru')  # Default to GRU as per paper
        input_dim = config.get('input_dim', 168)  # 168 hours per week
        num_features = config.get('num_features', 3)  # visit/outbound/stay
        hidden_dim = config.get('hidden_dim', 128)
        output_dim = config.get('output_dim', 128)
        
        total_input_dim = input_dim * num_features
        
        if self.encoder_type == 'gru':
            # GRU network for temporal mobility patterns (24×7 = 168 hours)
            # Input: [B, 168, 3] -> Output: [B, output_dim]
            self.gru = nn.GRU(
                input_size=num_features,
                hidden_size=hidden_dim,
                num_layers=config.get('num_layers', 2),
                batch_first=True,
                dropout=config.get('dropout', 0.1) if config.get('num_layers', 2) > 1 else 0
            )
            self.proj = nn.Linear(hidden_dim, output_dim)
        elif self.encoder_type == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        elif self.encoder_type == 'tcn':
            # Temporal Convolutional Network
            self.encoder = nn.Sequential(
                nn.Conv1d(num_features, hidden_dim, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}. Choose from 'gru', 'mlp', 'tcn'")
    
    def forward(self, mobility_data):
        """
        Args:
            mobility_data: [B, 168, 3] or [B, 168*3] - mobility features (24×7 hours, 3 features)
        Returns:
            encoded: [B, output_dim] - encoded mobility features
        """
        B = mobility_data.shape[0]
        
        if self.encoder_type == 'gru':
            # Ensure shape is [B, 168, 3] for GRU
            if len(mobility_data.shape) == 2:
                mobility_data = mobility_data.view(B, 168, 3)
            elif mobility_data.shape[1] != 168:
                # Reshape if needed: [B, 3, 168] -> [B, 168, 3]
                if mobility_data.shape[1] == 3 and mobility_data.shape[2] == 168:
                    mobility_data = mobility_data.transpose(1, 2)
                else:
                    # Flatten and reshape
                    mobility_data = mobility_data.view(B, -1)
                    mobility_data = mobility_data.view(B, 168, 3)
            
            # Normalize per time step
            denom = mobility_data.sum(dim=2, keepdim=True).clamp_min(1e-6)
            mobility_data = mobility_data / denom
            
            # GRU forward: [B, 168, 3] -> [B, 168, hidden_dim]
            gru_out, hidden = self.gru(mobility_data)
            # Use last hidden state: [B, hidden_dim]
            last_hidden = gru_out[:, -1, :]  # or use hidden[-1] for multi-layer
            # Project to output dimension
            encoded = self.proj(last_hidden)
            return encoded
        elif self.encoder_type == 'mlp':
            # Flatten if needed
            if len(mobility_data.shape) == 3:
                denom = mobility_data.sum(dim=1, keepdim=True).clamp_min(1e-6)
                mobility_data = mobility_data / denom
                mobility_data = mobility_data.view(B, -1)
            mobility_data = F.normalize(mobility_data, p=2, dim=1, eps=1e-6)
            return self.encoder(mobility_data)
        else:  # TCN
            # Ensure shape is [B, features, time]
            if len(mobility_data.shape) == 2:
                mobility_data = mobility_data.view(B, 3, 168)
            elif mobility_data.shape[1] != 3:
                mobility_data = mobility_data.transpose(1, 2)
            return self.encoder(mobility_data)

