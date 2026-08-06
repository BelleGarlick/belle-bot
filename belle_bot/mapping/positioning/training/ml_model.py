import torch
import torch.nn as nn

from belle_bot.mapping.positioning.training.models import ModalityEnum


device = torch.device('mps')


class PositionalModelling(nn.Module):
    def __init__(self, max_feature_dim, embed_dim):
        super().__init__()
        # Distinct projection layers for each modality
        self.imu_proj = nn.Linear(max_feature_dim, embed_dim)
        self.gps_proj = nn.Linear(max_feature_dim, embed_dim)

        # Learnable indicator embeddings for each modality type
        self.modality_embed = nn.Embedding(3, embed_dim)  # 0: pad, 1: imu, 2: gps

        self.lstm = nn.LSTM(
            embed_dim,
            embed_dim,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.05
        )
        self.out = nn.Linear(embed_dim, 3)

    def forward(self, merged_seq, modality_ids, hc=None, return_state=False):
        batch_size, seq_len, _ = merged_seq.shape

        imu_mask = torch.where(modality_ids == ModalityEnum.IMU, 1, 0).to(dtype=merged_seq.dtype, device=modality_ids.device).unsqueeze(-1)
        gps_mask = torch.where(modality_ids == ModalityEnum.GPS, 1, 0).to(dtype=merged_seq.dtype, device=modality_ids.device).unsqueeze(-1)

        # embed and create the various projections
        imu_projection = self.imu_proj(merged_seq)
        gps_projection = self.gps_proj(merged_seq)
        x = imu_projection * imu_mask + gps_projection * gps_mask

        # add modality embedding and time embedding
        x = x + self.modality_embed(modality_ids)

        # feed through the network
        x, hc = self.lstm(x, hc)
        x = self.out(x[:, -1, :])

        if return_state:
            return x, hc
        return x


class UnifiedSequenceTransformer(nn.Module):
    def __init__(self, sequence_length, max_feature_dim, embed_dim, nhead=16):
        super().__init__()
        # Distinct projection layers for each modality
        self.imu_proj = nn.Linear(max_feature_dim, embed_dim)
        self.gps_proj = nn.Linear(max_feature_dim, embed_dim)

        # Learnable indicator embeddings for each modality type
        self.modality_embed = nn.Embedding(3, embed_dim)  # 0: pad, 1: imu, 2: gps

        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(embed_dim, 3)

    def forward(self, merged_seq, modality_ids):
        batch_size, seq_len, _ = merged_seq.shape

        imu_mask = torch.where(modality_ids == ModalityEnum.IMU, 1, 0).to(dtype=merged_seq.dtype, device=modality_ids.device).unsqueeze(-1)
        gps_mask = torch.where(modality_ids == ModalityEnum.GPS, 1, 0).to(dtype=merged_seq.dtype, device=modality_ids.device).unsqueeze(-1)

        # embed and create the various projections
        imu_projection = self.imu_proj(merged_seq)
        gps_projection = self.gps_proj(merged_seq)
        x = imu_projection * imu_mask + gps_projection * gps_mask

        # add modality embedding and time embedding
        x = x + self.modality_embed(modality_ids)
        x = x + self.pos_embed

        # feed through the network
        x = self.transformer(x)
        x = self.out(torch.mean(x, dim=1))

        return x
