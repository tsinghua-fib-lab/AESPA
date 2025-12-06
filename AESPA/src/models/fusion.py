"""
Fusion Module with Cross-Attention and FiLM

Implements Section 3.2: Cross-Feature Fusion
- Image Cross Attention: Satellite features attend to Street View features
- FiLM-style Conditioner: Injects Proxy and Mobility vectors into visual features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM-style Conditioner)
    
    Injects Proxy and Mobility vectors into visual features:
    modulated_features = gamma(condition) * features + beta(condition)
    """
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)
    
    def forward(self, x, condition):
        """
        Args:
            x: [B, D] - features to modulate
            condition: [B, condition_dim] - conditioning vector
        Returns:
            modulated: [B, D] - modulated features
        """
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma * x + beta


class CrossAttentionLayer(nn.Module):
    """
    Image Cross Attention Layer
    
    Implements cross-attention between satellite and street view features.
    Satellite features (query) attend to street view features (key-value).
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, D] - query features (e.g., satellite)
            key_value: [B, D] - key-value features (e.g., street view)
        Returns:
            output: [B, D] - attended features
        """
        B = query.shape[0]
        residual = query
        
        # Multi-head attention
        Q = self.q_proj(query).view(B, self.num_heads, self.head_dim)
        K = self.k_proj(key_value).view(B, self.num_heads, self.head_dim)
        V = self.v_proj(key_value).view(B, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, V)
        attended = attended.view(B, self.hidden_dim)
        attended = self.out_proj(attended)
        
        # Residual connection and layer norm
        output = self.norm(attended + residual)
        return output


class AdaptiveGating(nn.Module):
    """
    Adaptive Gating for conditional tokens (Section 3.5)
    
    Forms conditioning vector u_i = [e_city(i) || p_i || z_mob]
    where:
    - e_city(i): learnable city embedding
    - p_i: proxy vector (5 physical proxies)
    - z_mob: mobility embedding (teacher only)
    """
    def __init__(self, token_dim, hidden_dim, num_cities=8):
        super().__init__()
        self.token_dim = token_dim
        self.num_cities = num_cities
        
        # Learnable city embedding (Section 3.5: e_city(i))
        self.city_embedding = nn.Embedding(num_cities, 8)  # 8 MSAs in paper
        
        self.token_embedding = nn.Linear(token_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, tokens):
        """
        Aggregate heterogeneous conditioning tensors into fixed length.
        Forms u_i = [e_city(i) || p_i || z_mob] as per Section 3.5
        """
        device = next(self.parameters()).device
        feature_list = []
        batch_size = None

        # Extract city_id and embed it (e_city(i))
        city_id = tokens.get('city_id', None)
        if city_id is not None:
            if isinstance(city_id, torch.Tensor):
                city_tensor = city_id.to(device).long()
            else:
                city_tensor = torch.tensor(city_id, dtype=torch.long, device=device)
            if city_tensor.dim() == 0:
                city_tensor = city_tensor.unsqueeze(0)
            city_emb = self.city_embedding(city_tensor)  # [B, 8]
            feature_list.append(city_emb)
            batch_size = city_emb.shape[0]

        # Add proxy vector (p_i) and mobility (z_mob) if available
        for key, value in tokens.items():
            if key == 'city_id':
                continue  # Already handled
            if isinstance(value, torch.Tensor):
                tensor = value.to(device)
            else:
                tensor = torch.tensor(value, dtype=torch.float32, device=device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            elif tensor.dim() > 2:
                tensor = tensor.view(tensor.shape[0], -1)
            feature_list.append(tensor)
            if batch_size is None:
                batch_size = tensor.shape[0]

        if not feature_list:
            if batch_size is None:
                batch_size = 1
            return torch.zeros(batch_size, self.token_embedding.out_features, device=device)

        token_matrix = torch.cat(feature_list, dim=1)
        feature_dim = token_matrix.shape[1]
        if feature_dim < self.token_dim:
            token_matrix = F.pad(token_matrix, (0, self.token_dim - feature_dim))
        elif feature_dim > self.token_dim:
            token_matrix = token_matrix[:, :self.token_dim]

        embedded = self.token_embedding(token_matrix)
        return self.gate(embedded)


class FusionModule(nn.Module):
    """Fusion Module with Cross-Attention and FiLM"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.get('hidden_dim', 512)
        num_layers = config.get('num_cross_attn_layers', 2)
        film_dim = config.get('film_dim', 128)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # Adaptive gating for conditional tokens (Section 3.5: u_i = [e_city(i) || p_i || z_mob])
        token_dim = config.get('token_dim', 13)
        num_cities = config.get('num_cities', 8)  # 8 MSAs in paper
        self.adaptive_gating = AdaptiveGating(token_dim, film_dim, num_cities=num_cities)
        
        # FiLM layers
        self.film = FiLM(film_dim, hidden_dim)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, street_features, satellite_features, conditional_tokens=None):
        """
        Cross-Feature Fusion with Image Cross Attention and FiLM
        
        Args:
            street_features: [B, D] - street view features
            satellite_features: [B, D] - satellite features
            conditional_tokens: dict - contains Proxy and Mobility vectors
                - Proxy: concept predictions (NDVI, tree, impervious, albedo, shadow)
                - Mobility: encoded mobility features (teacher only)
        Returns:
            fused_features: [B, D] - fused features
        """
        # Step 1: Image Cross Attention
        # Satellite (query) attends to Street View (key-value)
        attended_satellite = satellite_features
        for cross_attn in self.cross_attn_layers:
            attended_satellite = cross_attn(attended_satellite, street_features)
        
        # Step 2: Generate conditioning vector from Proxy and Mobility tokens
        if conditional_tokens is not None:
            condition = self.adaptive_gating(conditional_tokens)
        else:
            # Default zero condition
            B = street_features.shape[0]
            condition = torch.zeros(B, self.adaptive_gating.token_embedding.out_features,
                                  device=street_features.device)
        
        # Step 3: FiLM-style modulation
        # Inject Proxy and Mobility information into visual features
        modulated_satellite = self.film(attended_satellite, condition)
        modulated_street = self.film(street_features, condition)
        
        # Step 4: Concatenate and fuse
        concatenated = torch.cat([modulated_satellite, modulated_street], dim=-1)
        fused = self.fusion(concatenated)
        
        return fused

