import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from belle_bot.mapping.positioning.training.create_dataset import load_dataset, ImuData
from belle_bot.mapping.positioning.training.models import GpsPoint


device = torch.device('mps')


# realistically, i think the best thing to do is to jus use frames which embed the diff item
def transform_frame(x, tokens=100):
    first_gps_pos: GpsPoint = x.gps[0]
    x0, y0, alt0 = first_gps_pos.x, first_gps_pos.y, first_gps_pos.altitude

    # todo, take out the las tgps item
    # todo also make make it so that eventually we can make something more stochastic where frames are dropped / not perfectly discrete.
    # todo eventually change the first position change so instead we just predict relative point maybe
    # todo encode the distance how long before the entry is as a data in the frame relative to the point at which the prediction is to be made for

    output_gps = x.gps[-1]
    x.gps.remove(output_gps)

    all_data = sorted(x.gps + x.imu, key=lambda x: x.timestamp)
    all_data = all_data[-tokens:]

    modality_types = []
    modality_data = []

    for item in all_data:
        if isinstance(item, ImuData):
            modality_types.append(0)
            # todo change the angle to cos / sin so taht it's measured a lil better at the point hte loop happens
            modality_data.append(np.hstack((item.acc, item.gyro / 90, item.angle / 180)))

        elif isinstance(item, GpsPoint):
            modality_types.append(1)
            # tODO change the normalisationj. automate it
            gps_datum = np.array([
                (item.x - x0) / 30,
                (item.y - y0) / 30,
                (item.altitude - alt0)
            ] + [0] * 6)
            modality_data.append(gps_datum)  # padded the item so everything is same size

        else:
            # split up camera into multiple tokens
            raise NotImplementedError()

    if np.array(modality_data).min() < -100:
        breakpoint()

    return (
        np.array(modality_types),
        np.array(modality_data),
        [x0, y0, alt0],  # allows us to get true position
        [(output_gps.x - x0) / 30, (output_gps.y - y0) / 30, output_gps.altitude - alt0]  # allows us to estimate the final position
    )


def transform(x):
    modality_types = []
    modality_frames = []
    ys = []

    for item in x:
        modality_type, modality_data, _, y = transform_frame(item)
        modality_types.append(modality_type)
        modality_frames.append(modality_data)
        ys.append(y)

    # todo eventually make it so it's jagged

    return np.array(modality_types), np.array(modality_frames), np.array(ys)



class UnifiedSequenceTransformer(nn.Module):
    def __init__(self, sequence_length, max_feature_dim, embed_dim):
        super().__init__()
        # Distinct projection layers for each modality
        self.imu_proj = nn.Linear(max_feature_dim, embed_dim)
        self.gps_proj = nn.Linear(max_feature_dim, embed_dim)

        # Learnable indicator embeddings for each modality type (not used atm)
        self.modality_embed = nn.Embedding(1, embed_dim)  # 0: imu, 1: gps

        # Sequence Processor

        self.l1 = nn.Linear(sequence_length * embed_dim, sequence_length)
        self.l2 = nn.Linear(sequence_length, 3)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, merged_seq, modality_ids):
        batch_size, seq_len, _ = merged_seq.shape

        imu_mask = torch.where(modality_ids == 0, 0, 1).to(dtype=modality_ids.dtype, device=modality_ids.device).unsqueeze(-1)
        gps_mask = torch.where(modality_ids == 1, 0, 1).to(dtype=modality_ids.dtype, device=modality_ids.device).unsqueeze(-1)

        # embed and create the various projections
        imu_projection = self.imu_proj(merged_seq)
        gps_projection = self.gps_proj(merged_seq)
        combined_projection = imu_projection * imu_mask + gps_projection * gps_mask

        # feed through the network
        x = self.l1(combined_projection.flatten(1))
        x = F.relu(x)
        x = self.l2(x)
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
    train_x, val_x = load_dataset()
    train_x = transform(train_x)
    val_x = transform(val_x)

    model = UnifiedSequenceTransformer(100, 9, 5).to(device)
    optimizer = torch.optim.AdamW(model.parameters())

    # todo need to normalise

    # todo maybe pass each of the items in and a sequece order to pair them up in the model...

    for epoch in range(100):
        # todo batch
        epoch_loss = []
        for modality_types, modality_frames, ys in batchify(train_x):
            optimizer.zero_grad()

            prediction = model(modality_frames, modality_types)
            loss = F.mse_loss(prediction, ys)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        print(np.mean(epoch_loss))
