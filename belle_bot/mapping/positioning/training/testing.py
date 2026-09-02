import copy

import torch
import numpy as np

from belle_bot.mapping.positioning.training.environment import Episode
from belle_bot.mapping.positioning.training.environment.env import Frame
from belle_bot.mapping.positioning.training.environment.episode_processor import load_episodes
from belle_bot.mapping.positioning.training.environment.preprocessor import process_state
from belle_bot.mapping.positioning.training.ml_model import PositionalModelling
from belle_bot.mapping.positioning.training.models import GpsPoint
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds
from belle_bot.mapping.positioning.training.seeding import set_seed


MAX_SNAP_GPS_DISTANCE = 510


# todo change to in memory state to feed in one at a time


def perform_evals(
        episodes: list[Episode],
        model: PositionalModelling,
        bounds: NormalisationBounds,
        plot: bool = False
):
    device = torch.device('mps')

    position_errors = []
    final_position_errors = []

    model.eval()
    with torch.no_grad():
        for episode in episodes:
            true_positions = []
            predicted_positions = []
            
            # hc = None

            current_position = episode.current_position()
            data: list[Frame] = []
            while True:
                step = episode.step()

                pre_position_change = np.zeros((3,))
                if data:
                    pre_position_change = data[-1].position_change

                gt_position_change = step.new_position - step.last_position

                frame = copy.deepcopy(step.frame)
                if isinstance(frame, GpsPoint):
                    frame.x -= current_position[0]
                    frame.y -= current_position[1]
                    frame.altitude -= current_position[2]

                data.append(Frame(
                    frame=frame,
                    position=np.array(current_position),
                    prev_position_change=pre_position_change,
                    true_position=step.new_position,
                    prev_position=np.array(current_position),
                    position_change=np.zeros(3),
                    time_delta=step.delta_time
                ))

                modal_data_np, modal_types_np = process_state(data, 100, bounds)

                modal_data = torch.tensor(modal_data_np, dtype=torch.float32, device=device)
                modal_types = torch.tensor(modal_types_np, dtype=torch.int64, device=device)

                pred_delta = model(modal_data, modal_types).cpu().detach().numpy()[0] * MAX_SNAP_GPS_DISTANCE

                position_errors.append(np.linalg.norm(gt_position_change - pred_delta))

                current_position = current_position + pred_delta
                data[-1].position_change = pred_delta

                predicted_positions.append(np.array(current_position))
                true_positions.append(step.new_position)

                if step.terminated:
                    break

            final_position_errors.append(np.linalg.norm(current_position - step.new_position))

            if plot:
                import matplotlib
                matplotlib.use('macosx')
                import matplotlib.pyplot as plt
                plt.plot([x[0] for x in true_positions], [x[1] for x in true_positions])
                plt.plot([x[0] for x in predicted_positions], [x[1] for x in predicted_positions], '--')
                plt.axis("equal")
                plt.show()

    return {
        "mean_position_error": np.mean(position_errors),
        "mean_final_position_error": np.mean(final_position_errors),
    }


if __name__ == "__main__":
    set_seed(42)
    device = torch.device('mps')

    model = PositionalModelling(13, 256, n_layers=2).to(device)
    model.load_state_dict(torch.load("model-400000.pt"))

    perform_evals(
        episodes=load_episodes("training"),
        model=model,
        bounds=NormalisationBounds() \
            .load("bounds.json"),
        plot=True
    )