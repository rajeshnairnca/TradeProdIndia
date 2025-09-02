import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from . import config

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that combines a recurrent layer (LSTM) for time-series analysis
    with a Transformer layer for cross-sectional analysis among stocks.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int, num_sectors: int, sector_embed_dim: int):
        super().__init__(observation_space, features_dim)
        
        # The observation space for stock_features is now flattened by VecFrameStack
        # Original shape: (n_stocks, n_features)
        # After VecFrameStack: (n_stocks, n_features * n_stack)
        stock_features_shape = observation_space["stock_features"].shape
        global_features_dim = observation_space["global_features"].shape[-1] // config.LSTM_N_STACK
        
        # The feature dimension for a single stock at a single time step
        self.stock_feature_dim = stock_features_shape[-1] // config.LSTM_N_STACK
        
        self.num_sectors = num_sectors
        self.sector_embed_dim = sector_embed_dim
        
        # --- Recurrent Layer (LSTM) ---
        self.lstm_hidden_size = 64
        self.lstm = nn.LSTM(
            input_size=self.stock_feature_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        # --- Embedding Layers ---
        self.sector_embedding = nn.Embedding(num_embeddings=self.num_sectors, embedding_dim=self.sector_embed_dim)
        
        combined_stock_features_dim = self.lstm_hidden_size + self.sector_embed_dim
        self.stock_embedding = nn.Linear(combined_stock_features_dim, config.EMBED_DIM)
        self.embedding_dropout = nn.Dropout(0.2)
        
        # --- Transformer Layer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM, 
            nhead=2, 
            dim_feedforward=128, 
            dropout=0.3,
            activation='relu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # --- Final Projection ---
        self.final_dropout = nn.Dropout(0.2)
        self.final_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM + global_features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Deconstruct the observation dictionary
        stock_features = observations["stock_features"]
        sector_ids = observations["sector_ids"]
        global_features = observations["global_features"]
        mask = observations["mask"].to(torch.bool)

        batch_size, n_stocks, _ = stock_features.shape

        # Un-stack features concatenated by VecFrameStack
        # Shape (batch, n_stocks, features * n_stack) -> (batch, n_stocks, n_stack, features)
        unstacked_stock_features = stock_features.reshape(
            batch_size, n_stocks, config.LSTM_N_STACK, self.stock_feature_dim
        )

        # We only need the most recent version of non-stock-specific features
        # Shape (batch, n_stocks * n_stack) -> (batch, n_stocks)
        latest_sector_ids = sector_ids.reshape(batch_size, n_stocks, config.LSTM_N_STACK)[:, :, -1]
        latest_mask = mask.reshape(batch_size, n_stocks, config.LSTM_N_STACK)[:, :, -1]
        
        # Shape (batch, global_dim * n_stack) -> (batch, global_dim)
        global_features_dim = global_features.shape[-1] // config.LSTM_N_STACK
        latest_global_features = global_features.reshape(batch_size, config.LSTM_N_STACK, global_features_dim)[:, -1, :]

        # 1. LSTM Processing
        # Reshape for LSTM: (batch * n_stocks, n_stack, features)
        lstm_input = unstacked_stock_features.permute(0, 2, 1, 3).reshape(
            batch_size * n_stocks, config.LSTM_N_STACK, self.stock_feature_dim
        )
        
        _, (hidden_state, _) = self.lstm(lstm_input)
        
        time_aware_features = hidden_state[-1].reshape(batch_size, n_stocks, self.lstm_hidden_size)

        # 2. Transformer Processing
        clipped_ids = torch.clamp(latest_sector_ids.round(), 0, self.num_sectors - 1)
        sector_vecs = self.sector_embedding(clipped_ids.long())
        combined_features = torch.cat([time_aware_features, sector_vecs], dim=-1)
        
        embedded_stocks = self.stock_embedding(combined_features)
        embedded_stocks = self.embedding_dropout(embedded_stocks)

        padding_mask = ~latest_mask
        transformer_output = self.transformer_encoder(embedded_stocks, src_key_padding_mask=padding_mask)
        
        # 3. Final Aggregation
        masked_output = transformer_output.masked_fill(padding_mask.unsqueeze(-1), 0)
        sum_output = masked_output.sum(dim=1)
        num_valid_stocks = latest_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled_features = sum_output / num_valid_stocks
        
        final_features = torch.cat([mean_pooled_features, latest_global_features], dim=1)
        final_features = self.final_dropout(final_features)
        
        return self.final_projection(final_features)
