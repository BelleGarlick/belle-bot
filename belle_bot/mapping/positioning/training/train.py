import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from belle_bot.mapping.positioning.training.create_dataset import load_dataset, ImuData
from belle_bot.mapping.positioning.training.models import GpsPoint, GPSReplay
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds

device = torch.device('mps')

# realistically, i think the best thing to do is to jus use frames which embed the diff item
def transform_frame(x: GPSReplay, normaliser_bounds):
    first_gps_pos: GpsPoint = [item for item in x.events if isinstance(item, GpsPoint)][0]
    x0, y0, alt0 = first_gps_pos.x, first_gps_pos.y, first_gps_pos.altitude

    # todo notmalisation is currently based globally on the initial position. We need to make it rotation agnostic and position agnostic. e.g. if gps isn't given we still need to estimate the position and maybe do a delta pos rather than global
    # todo also make make it so that eventually we can make something more stochastic where frames are dropped / not perfectly discrete.
    # todo eventually change the first position change so instead we just predict relative point maybe
    # todo encode the distance how long before the entry is as a data in the frame relative to the point at which the prediction is to be made for
    # todo create a way to visualise the predicted position as the data streams in

    modality_types = []
    modality_data = []

    for item in x.events:
        if isinstance(item, ImuData):
            # Update the modality data
            # todo change the angle to cos / sin so that it's measured a lil better at the point from -180 to 180
            modality_types.append(0)
            modality_data.append(np.hstack((
                item.acc / normalisation_bounds['imu.acc'],
                item.gyro / normalisation_bounds['imu.gyro'],
                item.angle / normalisation_bounds['imu.angle']
            )))

        elif isinstance(item, GpsPoint):
            delta_x = item.x - x0
            delta_y = item.y - y0
            delta_alt = item.altitude - alt0

            # Update the modality data
            modality_types.append(1)
            modality_data.append(np.array([
                delta_x / normalisation_bounds["gps.x"],
                delta_y / normalisation_bounds["gps.y"],
                delta_alt / normalisation_bounds["gps.alt"],
            ] + [0] * 6))  # padded the item so everything is same size

        else:
            # split up camera into multiple tokens
            raise NotImplementedError()

    return (
        np.array(modality_types),
        np.array(modality_data),
        [x0, y0, alt0],  # allows us to get true position
        [
            (x.target.x - x0) / normalisation_bounds["gps.x"],
            (x.target.y - y0) / normalisation_bounds["gps.y"],
            (x.target.altitude - alt0) / normalisation_bounds["gps.alt"],
        ]  # allows us to estimate the final position
    )


def transform(x, normaliser_bounds):
    modality_types = []
    modality_frames = []
    ys = []

    for item in x:
        modality_type, modality_data, _, y = transform_frame(item, normaliser_bounds)

        # todo eventually make it so it's jagged by adding padding rather than dropping
        if len(modality_type) != 100:
            continue

        modality_types.append(modality_type)
        modality_frames.append(modality_data)
        ys.append(y)


    return np.array(modality_types), np.array(modality_frames), np.array(ys)



class UnifiedSequenceTransformer(nn.Module):
    def __init__(self, sequence_length, max_feature_dim, embed_dim):
        super().__init__()
        # Distinct projection layers for each modality
        self.imu_proj = nn.Linear(max_feature_dim, embed_dim)
        self.gps_proj = nn.Linear(max_feature_dim, embed_dim)

        # Learnable indicator embeddings for each modality type (not used atm)
        self.modality_embed = nn.Embedding(1, embed_dim)  # 0: imu, 1: gps
        # todo also need the time embedding

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(embed_dim, 3)

    def forward(self, merged_seq, modality_ids):
        batch_size, seq_len, _ = merged_seq.shape

        imu_mask = torch.where(modality_ids == 0, 0, 1).to(dtype=modality_ids.dtype, device=modality_ids.device).unsqueeze(-1)
        gps_mask = torch.where(modality_ids == 1, 0, 1).to(dtype=modality_ids.dtype, device=modality_ids.device).unsqueeze(-1)

        # embed and create the various projections
        imu_projection = self.imu_proj(merged_seq)
        gps_projection = self.gps_proj(merged_seq)
        combined_projection = imu_projection * imu_mask + gps_projection * gps_mask

        # todo add modality embedding and time embedding

        # feed through the network
        x = self.transformer(combined_projection)
        x = self.out(x[:, -1])
        return x


def batchify(x, batch_size=16):
    idxs = np.arange(len(x[0]))
    np.random.shuffle(idxs)

    items = []
    for i in range(0, len(x[0]), batch_size):
        start, end = i, i + batch_size
        items.append((
            torch.tensor(x[0][idxs[start:end]], dtype=torch.float, device=device),
            torch.tensor(x[1][idxs[start:end]], dtype=torch.float, device=device),
            torch.tensor(x[2][idxs[start:end]], dtype=torch.float, device=device),
        ))

    return items


if __name__ == "__main__":
    train_replays, val_replays = load_dataset()

    normalisation_bounds = NormalisationBounds().fit(train_replays)

    model = UnifiedSequenceTransformer(100, 9, 20).to(device)
    optimizer = torch.optim.AdamW(model.parameters())

    train_x = transform(train_replays, normalisation_bounds)
    val_x = transform(val_replays, normalisation_bounds)

    # todo maybe pass each of the items in and a sequece order to pair them up in the model...
    for epoch in range(200):
        # Perform training
        model.train()
        train_epoch_loss = []
        for modality_types, modality_frames, ys in batchify(train_x):
            optimizer.zero_grad()

            prediction = model(modality_frames, modality_types)
            loss = F.mse_loss(prediction, ys)

            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())

        # Run validation
        val_epoch_loss = []
        model.eval()
        with torch.no_grad():
            for modality_types, modality_frames, ys in batchify(val_x):
                optimizer.zero_grad()

                prediction = model(modality_frames, modality_types)
                loss = F.mse_loss(prediction, ys)

                val_epoch_loss.append(loss.item())

        print(np.mean(train_epoch_loss), np.mean(val_epoch_loss))
