import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from belle_bot.mapping.positioning.training.create_dataset import load_dataset, ImuData
from belle_bot.mapping.positioning.training.models import GpsPoint, GPSReplay
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds

SEQUENCE_LENGTH = 25

device = torch.device('mps')

# realistically, i think the best thing to do is to jus use frames which embed the diff item
def transform_frame(x: GPSReplay, normalisation_bounds):
    first_gps_pos: GpsPoint = [item for item in x.events if isinstance(item, GpsPoint)][0]
    x0, y0, alt0 = first_gps_pos.x, first_gps_pos.y, first_gps_pos.altitude

    # todo notmalisation is currently based globally on the initial position. We need to make it rotation agnostic and position agnostic. e.g. if gps isn't given we still need to estimate the position and maybe do a delta pos rather than global
    # todo also make make it so that eventually we can make something more stochastic where frames are dropped / not perfectly discrete.
    # todo eventually change the first position change so instead we just predict relative point maybe
    # todo visualise the frame
    # todo encode the distance how long before the entry is as a data in the frame relative to the point at which the prediction is to be made for

    modality_types = []
    modality_data = []
    time_deltas = []

    target_ts = x.target.timestamp if x.target is not None else 0

    for item in x.events:
        time_deltas.append(item.timestamp - target_ts)
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

    target = None
    if x.target is not None:
        target = [
            (x.target.x - x0) / normalisation_bounds["gps.x"],
            (x.target.y - y0) / normalisation_bounds["gps.y"],
            (x.target.altitude - alt0) / normalisation_bounds["gps.alt"],
        ]

    return (
        np.array(modality_types),
        np.array(modality_data),
        np.array(time_deltas),
        [x0, y0, alt0],  # allows us to get true position
        target
    )


def transform(x, normaliser_bounds):
    modality_types = []
    modality_frames = []
    time_deltas = []
    ys = []

    for item in x:
        modality_type, modality_data, delta_t, _, y = transform_frame(item, normaliser_bounds)

        # todo eventually make it so it's jagged by adding padding rather than dropping
        if len(modality_type) != SEQUENCE_LENGTH:
            continue

        modality_types.append(modality_type)
        modality_frames.append(modality_data)
        time_deltas.append(delta_t)
        ys.append(y)

    return np.array(modality_types), np.array(modality_frames), np.array(time_deltas), np.array(ys)



class UnifiedSequenceTransformer(nn.Module):
    def __init__(self, sequence_length, max_feature_dim, embed_dim, nhead=16):
        super().__init__()
        # Distinct projection layers for each modality
        self.imu_proj = nn.Linear(max_feature_dim, embed_dim)
        self.gps_proj = nn.Linear(max_feature_dim, embed_dim)

        # Learnable indicator embeddings for each modality type
        self.modality_embed = nn.Embedding(2, embed_dim)  # 0: imu, 1: gps
        self.time_proj = nn.Linear(1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(embed_dim, 3)

    def forward(self, merged_seq, modality_ids, time_deltas):
        batch_size, seq_len, _ = merged_seq.shape

        imu_mask = torch.where(modality_ids == 0, 1, 0).to(dtype=merged_seq.dtype, device=modality_ids.device).unsqueeze(-1)
        gps_mask = torch.where(modality_ids == 1, 1, 0).to(dtype=merged_seq.dtype, device=modality_ids.device).unsqueeze(-1)

        # embed and create the various projections
        imu_projection = self.imu_proj(merged_seq)
        gps_projection = self.gps_proj(merged_seq)
        x = imu_projection * imu_mask + gps_projection * gps_mask

        # add modality embedding and time embedding
        # x = x + self.modality_embed(modality_ids)

        # Normalize time_deltas to roughly N(0, 1) scale so it doesn't overpower feature embeddings
        time_deltas_norm = (time_deltas - time_deltas.mean()) / (time_deltas.std() + 1e-6)
        x = x + self.time_proj(time_deltas_norm.unsqueeze(-1))

        # feed through the network
        x = self.transformer(x)
        x = self.out(torch.mean(x, dim=1))

        return x


def batchify(x, batch_size=256):
    idxs = np.arange(len(x[0]))
    np.random.shuffle(idxs)

    items = []
    for i in range(0, len(x[0]), batch_size):
        start, end = i, i + batch_size
        items.append((
            torch.tensor(x[0][idxs[start:end]], dtype=torch.long, device=device),
            torch.tensor(x[1][idxs[start:end]], dtype=torch.float, device=device),
            torch.tensor(x[2][idxs[start:end]], dtype=torch.float, device=device),
            torch.tensor(x[3][idxs[start:end]], dtype=torch.float, device=device),
        ))

    return items


if __name__ == "__main__":
    train_replays, val_replays = load_dataset(seq_len=SEQUENCE_LENGTH)

    normalisation_bounds = NormalisationBounds()\
        .fit(train_replays)\
        .save("bounds.json")

    model = UnifiedSequenceTransformer(SEQUENCE_LENGTH, 9, 256, nhead=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters())

    train_x = transform(train_replays, normalisation_bounds)
    val_x = transform(val_replays, normalisation_bounds)

    # todo maybe pass each of the items in and a sequece order to pair them up in the model...
    for epoch in range(200):
        # Perform training
        model.train()
        train_epoch_loss = []
        for modality_types, modality_frames, time_deltas, ys in batchify(train_x):
            optimizer.zero_grad()

            prediction = model(modality_frames, modality_types, time_deltas)
            loss = F.mse_loss(prediction, ys)

            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())

        # Run validation
        val_epoch_loss = []
        model.eval()
        with torch.no_grad():
            for modality_types, modality_frames, time_deltas, ys in batchify(val_x):
                optimizer.zero_grad()

                prediction = model(modality_frames, modality_types, time_deltas)
                loss = F.mse_loss(prediction, ys)

                val_epoch_loss.append(loss.item())

        print("Epoch", epoch, np.mean(train_epoch_loss), np.mean(val_epoch_loss))

    torch.save(model.state_dict(), "model.pt")
