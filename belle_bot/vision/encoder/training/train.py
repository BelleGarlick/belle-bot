import math
from collections import deque
from contextlib import nullcontext

import matplotlib.pyplot as plt
import mlflow
import numpy as np

import torch

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.encoder.training.export_onnx import export_onnx
from belle_bot.vision.encoder.training.data_loader import load_dataset
from belle_bot.vision.encoder.training.loss import OptimizedVaeLoss
from belle_bot.vision.encoder.training.ml_model import VAE2_448

from torch.utils.data import DataLoader

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

config = clpy.parse_cli_args(VisionEncoderTrainingConfig())

model = VAE2_448(latent_dim=config.model.embedding_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
loss_fn = OptimizedVaeLoss().to(DEVICE)

train_dataset, test_dataset = load_dataset(config)
pin_memory = DEVICE.type != 'mps'
train_loader = DataLoader(train_dataset, batch_size=None, num_workers=4, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=None, num_workers=2, pin_memory=pin_memory)

# todo
#  mask out 0 depth data items so the model has to learn how to infer them
#  have a fully trained model and explore random generation
#  possible have the encoder trained via a multi-task embedding
#  possibly drop input data and have the model try and predict both
#  have an automated pipeline for tagging items if their error is high enough

raise Exception("future work should ranodmly mask out parts of the input depth")

def sample_model(step, train_batch, test_batch):
    cm = plt.get_cmap('rainbow')

    def render_depth(d):
        d = d / (np.max(d) + 1e-8)
        return cm(d)[:, :, :3]

    model.eval()
    with torch.no_grad():
        # Get one batch from train and test
        for name, batch in [("train", train_batch), ("test", test_batch)]:
            recon, _, _ = model(batch)
            
            y = recon.detach().cpu().numpy()
            x_np = batch.detach().cpu().numpy()

            image_pairs = []
            for i in range(min(8, x_np.shape[0])):
                input_rgb = x_np[i, :3, :, :].transpose((1, 2, 0))
                input_depth = render_depth(x_np[i, 3, :, :])
                output_rgb = y[i, :3, :, :].transpose((1, 2, 0))
                output_depth = render_depth(y[i, 3, :, :])

                # Create the image grid for a single item
                pair = np.vstack((
                    np.hstack((input_rgb, input_depth)),
                    np.hstack((output_rgb, output_depth))
                ))
                image_pairs.append(pair)

            # Pair all the items together
            row = np.vstack((
                np.hstack(image_pairs[:4]),
                np.hstack(image_pairs[4:]),
            ))

            plt.figure(figsize=(32, 16))
            plt.title(f"Reconstructions ({name}) - Left: RGB, Right: Depth | Top: Input, Bottom: Output")
            plt.imshow(np.clip(row, 0, 1))
            plt.savefig(f"{name}_{step}.png")
            plt.close()


def mean_loss(data):
    return {
        "loss": np.mean([x['loss'].item() for x in data]),
        "recon_loss": np.mean([x['recon_loss'].item() for x in data]),
        "rgb_loss": np.mean([x['rgb_loss'].item() for x in data]),
        "depth_loss": np.mean([x['depth_loss'].item() for x in data]),
        "perc_loss": np.mean([x['perc_loss'].item() for x in data]),
        "kl_loss": np.mean([x['kl_loss'].item() for x in data]),
    }


if __name__ == "__main__":
    # We use the existing iterator for validation to avoid re-initializing
    last_test_batch = None
    best_val_loss = math.inf

    train_dl = iter(train_loader)
    test_dl = iter(test_loader)

    if config.mlflow.enabled:
        mlflow.set_tracking_uri(config.mlflow.endpoint)
        mlflow.set_experiment("vision-encoder")
        mlflow_run = mlflow.start_run(run_name=str("optimised-vae"))
    else:
        mlflow_run = nullcontext()

    with mlflow_run:
        if config.mlflow.enabled:
            mlflow.log_params(clpy.to_dict(config))

        epoch_loss_train = deque(maxlen=2000)
        for step, batch in enumerate(train_dl):
            if step >= config.max_steps:
                break
            
            batch = batch.to(DEVICE)
            model.to(DEVICE)

            percentage_complete = step / config.max_steps
            percentage_remaining = 1 - percentage_complete

            # scale the learning rate through training
            scale = 1 - math.pow(percentage_remaining, config.learning_rate_gamma)
            for g in optimizer.param_groups:
                g['lr'] = config.learning_rate * scale

            model.train()
            optimizer.zero_grad()

            recon_images, mu, logvar = model(batch)
            # KL Annealing: start at 0, increase to 1
            kl_beta = percentage_complete * config.kl_annealing
            loss = loss_fn(recon_images, batch, mu, logvar, kl_beta=kl_beta)

            # Train the model
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_train.append(loss)
            print(f"\rStep: {step}. Loss: {mean_loss(epoch_loss_train)['loss']:.5f}", end="")

            # Render images if the step % is reached
            if (step + 1) % config.render_image_every_n_steps == 0 and last_test_batch is not None:
                sample_model(step, batch, last_test_batch)

            # Validation
            if (step + 1) % config.eval_every_n_steps == 0:
                epoch_loss_val = []

                model.eval()
                with torch.no_grad():
                    for val_step, batch_val in enumerate(test_dl):
                        if val_step >= config.max_val_steps:
                            break
                        batch_val = batch_val.to(DEVICE)
                        model.to(DEVICE)
                        last_test_batch = batch_val

                        recon_images, mu, logvar = model(batch_val)
                        kl_beta = min(1.0, step / (config.max_steps * 0.1)) * config.kl_annealing
                        loss = loss_fn(recon_images, batch_val, mu, logvar, kl_beta=kl_beta)

                        epoch_loss_val.append(loss)

                        print(f"\rValidating. Items: {val_step * config.mini_batch_size}. Loss: {mean_loss(epoch_loss_val)['loss']:.5f}", end="")

                    print(f"\rStep: {step}. Running Loss: {mean_loss(epoch_loss_train)['loss']:.5f} Val Loss: {mean_loss(epoch_loss_val)['loss']:.5f}")

                    # Log the metrics if enabled
                    if config.mlflow.enabled:
                        mlflow.log_metrics({
                            **mean_loss(epoch_loss_train),
                            **{f"val_{key}": value for key, value in mean_loss(epoch_loss_val).items()}
                        }, step=step + 1)

                    # if the model loss is the new best then save the model
                    current_step_loss = mean_loss(epoch_loss_val)['loss']
                    if current_step_loss < best_val_loss:
                        best_val_loss = current_step_loss

                        model_path = f"{config.checkpoint_name}.pt"
                        torch.save(model.state_dict(), model_path)
                        export_onnx(model, "model.onnx")
