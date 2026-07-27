import os

import numpy as np
import torch

from belle_bot.mapping.positioning.training.create_dataset import open_replay_file, parse_data
from belle_bot.mapping.positioning.training.models import GpsPoint, GPSReplay, ImuData, CameraData
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds
from belle_bot.mapping.positioning.training.train import transform_frame, UnifiedSequenceTransformer, device, \
    SEQUENCE_LENGTH

# predict position and confidence
# need to predict with missing data throughout the input



REPLAY_FILE_PATH = "/Users/belle/Developer/belle-bot/replays"
replay_files = os.listdir(REPLAY_FILE_PATH)


VALID_FILES = {
    "14240eda-3dfc-48e0-921b-c22397ca0981.txt",
    "5ba31b5c-b457-4c80-9dc4-2f42d9f61a9f.txt",
    # "87d554ee-3d54-4440-85cb-756f0c234067.txt"
}


if __name__ == "__main__":
    normalisation_bounds = NormalisationBounds().load("bounds.json")
    model = UnifiedSequenceTransformer(SEQUENCE_LENGTH, 9, 256, nhead=32).to(device)
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.eval()

    for replay_file in replay_files:
        if replay_file not in VALID_FILES: continue
        if replay_file[0] == ".": continue
        lines = open_replay_file(replay_file)

        replay_file = parse_data(lines)

        predicted_positions = []
        real_positiosn = []

        current_window = []
        for event in replay_file.events:
            if isinstance(event, GpsPoint):
                real_positiosn.append((event.x, event.y))

            current_window.append(event)
            current_window = current_window[-SEQUENCE_LENGTH:]
            if len(current_window) != SEQUENCE_LENGTH:
                continue

            modality_type, modality_frames, times, relative_pos, _ = transform_frame(GPSReplay(current_window), normalisation_bounds)

            modality_types = torch.tensor(np.expand_dims(modality_type, 0), dtype=torch.long, device=device)
            modality_frames = torch.tensor(np.expand_dims(modality_frames, 0), dtype=torch.float, device=device)
            times = torch.tensor(np.expand_dims(times, 0), dtype=torch.float, device=device)

            prediction = model(modality_frames, modality_types, times)
            prediction = prediction.cpu().detach().numpy()[0]
            prediction = relative_pos + np.array([
                prediction[0] * normalisation_bounds['gps.x'],
                prediction[1] * normalisation_bounds['gps.y'],
                prediction[2] * normalisation_bounds['gps.alt'],
            ])

            predicted_positions.append(prediction)

        predicted_positions = np.array(predicted_positions)
        real_positiosn = np.array(real_positiosn)

        import matplotlib
        matplotlib.use('macosx')
        import matplotlib.pyplot as plt

        plt.plot(predicted_positions[:, 0], predicted_positions[:, 1])
        plt.scatter(real_positiosn[:, 0], real_positiosn[:, 1], s=2, c='green')
        plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], s=2, c='red')
        plt.axis("equal")
        plt.show()

        # breakpoint()
