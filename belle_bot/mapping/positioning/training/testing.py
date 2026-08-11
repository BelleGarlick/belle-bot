import torch
import numpy as np

from belle_bot.mapping.positioning.training.environment.multi_environment import MultiEnvironment
from belle_bot.mapping.positioning.training.train import MAX_SNAP_GPS_DISTANCE, process_state
from belle_bot.mapping.positioning.training.ml_model import PositionalModelling
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds

# todo add heuristic to sampling states
# todo add learning rate decayed based on the step number
device = torch.device('mps')


# todo change to in memory state to feed in one at a time

if __name__ == "__main__":
    bounds = NormalisationBounds()\
        .load("bounds.json")

    model = PositionalModelling(13, 512, n_layers=2).to(device)
    model.load_state_dict(torch.load("model-50000.pt"))

    env = MultiEnvironment(None, seq_len=100, envs=1)

    with torch.no_grad():
        for i in range(len(env.replay_ids)):
            model.eval()

            reset_data = env.reset()
            state = reset_data.initial_states[0]

            # Start sample from the environment
            episode_step_error = []
            episode_losses = []

            true_positions = []
            predicted_positions = []

            hc = None
            terminated = False
            while not terminated:
                modal_data, modal_types = process_state(state, seq_length=1, normalisation_bounds=bounds)

                # Predict new position
                predicted_position_change, hc = model(
                    torch.tensor(modal_data).to(device=device, dtype=torch.float32),
                    torch.tensor(modal_types).to(device=device, dtype=torch.int64),
                    hc=hc,
                    return_state=True,
                )
                predicted_position_change = predicted_position_change.detach().cpu().numpy()[0] * MAX_SNAP_GPS_DISTANCE

                state, position_change, terminated = env.step(0, predicted_position_change)

                # To prevent errors exploding during instances where the drift increases, if the error is too high,
                # then we apply the position fix and effectively reset the trajectory
                # Doing so means we can keep training during this run
                step_error = np.linalg.norm(position_change - predicted_position_change)
                episode_step_error += [step_error]

                # position += predicted_position_change

            # import matplotlib
            # matplotlib.use('macosx')
            # import matplotlib.pyplot as plt
            # plt.plot([x[0] for x in true_positions], [x[1] for x in true_positions])
            # plt.plot([x[0] for x in predicted_positions], [x[1] for x in predicted_positions])
            # plt.axis("equal")
            # plt.show()

            print(env.replay_ids[i], np.mean(episode_step_error))