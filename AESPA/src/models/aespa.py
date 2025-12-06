"""
AESPA Model: Integrates encoders, fusion, and prediction heads

Main model class that supports:
- Teacher mode: Uses mobility encoder + all modalities
- Student mode: Visual-only (satellite + street view), no mobility

Output Heads:
- LST Prediction Head: Temperature regression
- Physics Proxy Head: Concept bottleneck for weak supervision
"""
import torch
import torch.nn as nn
from .encoders import StreetViewEncoder, SatelliteEncoder, MobilityEncoder
from .fusion import FusionModule


class ConceptBottleneck(nn.Module):
    """Concept Bottleneck Head for weak supervision"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.get('hidden_dim', 256)
        concepts = config.get('concepts', ['ndvi_proxy', 'tree_canopy_proxy', 'impervious_proxy', 'albedo_proxy', 'shadow_fraction'])
        
        self.concepts = concepts
        self.num_concepts = len(concepts)
        
        # Concept predictors
        self.concept_predictors = nn.ModuleDict()
        for concept in concepts:
            self.concept_predictors[concept] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def forward(self, features):
        """
        Args:
            features: [B, D] - input features
        Returns:
            concept_logits: dict - logits for each concept
        """
        concept_logits = {}
        for concept in self.concepts:
            logit = self.concept_predictors[concept](features)
            concept_logits[concept] = logit.squeeze(-1)
        return concept_logits


class LSTHead(nn.Module):
    """LST Regression Head"""
    def __init__(self, input_dim, hidden_dim=256, num_outputs=2):
        """
        Args:
            input_dim: input feature dimension
            hidden_dim: hidden dimension
            num_outputs: number of outputs (2 for day/night)
        """
        super().__init__()
        self.num_outputs = num_outputs
        
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_outputs)
        )
        
        # Initialize last layer bias to zero
        if num_outputs == 1:
            nn.init.zeros_(self.regression_head[-1].bias)
        else:
            for i in range(num_outputs):
                nn.init.zeros_(self.regression_head[-1].bias[i])
    
    def forward(self, features):
        """
        Args:
            features: [B, D] - input features
        Returns:
            lst_pred: [B, num_outputs] - predicted LST (day, night)
        """
        lst_pred = self.regression_head(features)
        return lst_pred


class TeacherModel(nn.Module):
    """Teacher Model with visual features + mobility"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        hidden_dim = model_config.get('hidden_dim', 512)
        
        # Encoders
        self.street_view_encoder = StreetViewEncoder(config.get('street_view', {}))
        self.satellite_encoder = SatelliteEncoder(config.get('satellite', {}))
        
        # Mobility encoder (teacher only)
        if config.get('mobility', {}).get('enabled', True):
            self.mobility_encoder = MobilityEncoder(config.get('mobility', {}))
        else:
            self.mobility_encoder = None
        
        concept_cfg = config.get('concept_bottleneck', {})
        if concept_cfg.get('enabled', True):
            concept_cfg = concept_cfg.copy()
            concept_cfg['hidden_dim'] = hidden_dim
            self.concept_bottleneck = ConceptBottleneck(concept_cfg)
            self.concept_names = list(self.concept_bottleneck.concepts)
        else:
            self.concept_bottleneck = None
            self.concept_names = concept_cfg.get('concepts', self.config.get('concept_names', []))

        concept_dim = len(self.concept_names) if self.concept_names else 0
        mobility_dim = 0
        if self.mobility_encoder is not None:
            mobility_dim = config.get('mobility', {}).get('output_dim', 128)

        # Fusion module
        fusion_config = {
            'hidden_dim': hidden_dim,
            'num_cross_attn_layers': model_config.get('num_cross_attn_layers', 2),
            'film_dim': model_config.get('film_dim', 128),
            'token_dim': 1 + 4 + 1 + 2 + concept_dim + mobility_dim
        }
        self.fusion = FusionModule(fusion_config)
        
        # LST regression head (predict day temperature)
        num_outputs = model_config.get('num_outputs', 1)  # Default to 1 (day only)
        self.lst_head = LSTHead(hidden_dim, hidden_dim=256, num_outputs=num_outputs)
        
        # Night temperature auxiliary head for ranking constraint
        self.use_night_aux = model_config.get('use_night_aux', False)
        if self.use_night_aux:
            self.night_head = LSTHead(hidden_dim, hidden_dim=256, num_outputs=1)
    
    def forward(self, street_images, satellite_image, mobility_data=None, 
                conditional_tokens=None, return_concepts=False):
        """
        Args:
            street_images: [B, K, C, H, W] - street view images
            satellite_image: [B, C, H, W] - satellite image
            mobility_data: [B, 168, 3] - mobility data (optional)
            conditional_tokens: dict - conditional tokens
            return_concepts: bool - whether to return concept predictions
        Returns:
            lst_pred: [B, 1] - predicted LST (day only)
            concepts: dict - concept predictions (if return_concepts=True)
            features: [B, D] - intermediate features (for distillation)
        """
        # Encode street view
        street_features, _ = self.street_view_encoder(street_images)
        
        # Encode satellite
        satellite_features = self.satellite_encoder(satellite_image)
        
        if conditional_tokens is None:
            conditional_tokens = {}
        else:
            conditional_tokens = {k: v for k, v in conditional_tokens.items()}

        if self.mobility_encoder is not None and mobility_data is not None:
            mobility_features = self.mobility_encoder(mobility_data)
            conditional_tokens['mobility'] = mobility_features
        
        # Fuse features
        fused_features = self.fusion(street_features, satellite_features, conditional_tokens)
        
        # Predict concepts
        concepts = None
        if self.concept_bottleneck is not None:
            concepts = self.concept_bottleneck(fused_features)
        
        # Predict LST (day temperature)
        lst_pred = self.lst_head(fused_features)
        
        # Predict night temperature (auxiliary, for ranking constraint)
        night_pred = None
        if self.use_night_aux:
            night_pred = self.night_head(fused_features)
        
        if return_concepts:
            return lst_pred, concepts, fused_features, night_pred
        return lst_pred, fused_features, night_pred


class StudentModel(nn.Module):
    """Student Model with visual features only (no mobility)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        hidden_dim = model_config.get('hidden_dim', 512)
        
        # Encoders
        self.street_view_encoder = StreetViewEncoder(config.get('street_view', {}))
        self.satellite_encoder = SatelliteEncoder(config.get('satellite', {}))

        concept_cfg = config.get('concept_bottleneck', {})
        if concept_cfg.get('enabled', True):
            concept_cfg = concept_cfg.copy()
            concept_cfg['hidden_dim'] = hidden_dim
            self.concept_bottleneck = ConceptBottleneck(concept_cfg)
            self.concept_names = list(self.concept_bottleneck.concepts)
        else:
            self.concept_bottleneck = None
            self.concept_names = concept_cfg.get('concepts', self.config.get('concept_names', []))

        concept_dim = len(self.concept_names) if self.concept_names else 0

        # Fusion module
        fusion_config = {
            'hidden_dim': hidden_dim,
            'num_cross_attn_layers': model_config.get('num_cross_attn_layers', 2),
            'film_dim': model_config.get('film_dim', 128),
            'token_dim': 1 + 4 + 1 + 2 + concept_dim
        }
        self.fusion = FusionModule(fusion_config)
        
        # LST regression head (predict day temperature)
        num_outputs = model_config.get('num_outputs', 1)  # Default to 1 (day only)
        self.lst_head = LSTHead(hidden_dim, hidden_dim=256, num_outputs=num_outputs)
        
        # Night temperature auxiliary head for ranking constraint
        self.use_night_aux = model_config.get('use_night_aux', False)
        if self.use_night_aux:
            self.night_head = LSTHead(hidden_dim, hidden_dim=256, num_outputs=1)
    
    def forward(self, street_images, satellite_image, conditional_tokens=None, 
                return_concepts=False):
        """
        Args:
            street_images: [B, K, C, H, W] - street view images
            satellite_image: [B, C, H, W] - satellite image
            conditional_tokens: dict - conditional tokens
            return_concepts: bool - whether to return concept predictions
        Returns:
            lst_pred: [B, 1] - predicted LST (day only)
            concepts: dict - concept predictions (if return_concepts=True)
            features: [B, D] - intermediate features (for distillation)
        """
        # Encode street view
        street_features, _ = self.street_view_encoder(street_images)
        
        # Encode satellite
        satellite_features = self.satellite_encoder(satellite_image)
        
        if conditional_tokens is None:
            conditional_tokens = {}
        else:
            conditional_tokens = {k: v for k, v in conditional_tokens.items()}
        
        # Fuse features (no mobility)
        fused_features = self.fusion(street_features, satellite_features, conditional_tokens)
        
        # Predict concepts
        concepts = None
        if self.concept_bottleneck is not None:
            concepts = self.concept_bottleneck(fused_features)
        
        # Predict LST (day temperature)
        lst_pred = self.lst_head(fused_features)
        
        # Predict night temperature (auxiliary, for ranking constraint)
        night_pred = None
        if self.use_night_aux:
            night_pred = self.night_head(fused_features)
        
        if return_concepts:
            return lst_pred, concepts, fused_features, night_pred
        return lst_pred, fused_features, night_pred

