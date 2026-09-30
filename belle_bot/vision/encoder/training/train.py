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


def sample_model(step, train_samples, test_samples, target_count=16):
    """
    Renders up to `target_count` samples in a dynamic grid layout.
    
    `train_samples` and `test_samples` are lists/deques of batch tensors.
    """
    cm = plt.get_cmap('rainbow')

    def render_depth(d):
        d = d / (np.max(d) + 1e-8)
        return cm(d)[:, :, :3]

    model.eval()
    with torch.no_grad():
        for name, samples_deque in [("train", train_samples), ("test", test_samples)]:
            if not samples_deque:
                continue

            # Concatenate collected batches into a single tensor up to target_count
            batch = torch.cat(list(samples_deque), dim=0)[:target_count].to(DEVICE)

            recon, _, _ = model(batch)

            y = recon.detach().cpu().numpy()
            x_np = batch.detach().cpu().numpy()

            image_pairs = []
            for i in range(x_np.shape[0]):
                input_rgb = x_np[i, :3, :, :].transpose((1, 2, 0))
                input_depth = render_depth(x_np[i, 3, :, :])
                output_rgb = y[i, :3, :, :].transpose((1, 2, 0))
                output_depth = render_depth(y[i, 3, :, :])

                # Create the image grid for a single item (Top: Input RGB+Depth, Bottom: Output RGB+Depth)
                pair = np.vstack((
                    np.hstack((input_rgb, input_depth)),
                    np.hstack((output_rgb, output_depth))
                ))
                image_pairs.append(pair)

            total_pairs = len(image_pairs)
            if total_pairs == 0:
                continue

            # Arrange into rows dynamically (up to 4 items per row)
            items_per_row = 4
            rows = []
            for r in range(0, total_pairs, items_per_row):
                row_items = image_pairs[r:r + items_per_row]
                if row_items:
                    rows.append(np.hstack(row_items))

            grid = np.vstack(rows)

            plt.figure(figsize=(32, 8 * len(rows)))
            plt.title(f"Reconstructions ({name}) - Left: RGB, Right: Depth | Top: Input, Bottom: Output")
            plt.imshow(np.clip(grid, 0, 1))
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
    best_val_loss = math.inf

    train_dl = iter(train_loader)
    test_dl = iter(test_loader)

    # Deques to buffer up to 16 samples for train and test visualizations
    TARGET_SAMPLE_COUNT = 16
    train_sample_buffer = deque()
    test_sample_buffer = deque()
    train_buffer_count = 0
    test_buffer_count = 0

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

            # Buffer training samples (detach & cpu to save GPU memory)
            if train_buffer_count < TARGET_SAMPLE_COUNT:
                train_sample_buffer.append(batch.detach().cpu())
                train_buffer_count += batch.shape[0]

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
            if (step + 1) % config.render_image_every_n_steps == 0 and len(test_sample_buffer) > 0:
                sample_model(step, train_sample_buffer, test_sample_buffer, target_count=TARGET_SAMPLE_COUNT)

            # Validation
            if (step + 1) % config.eval_every_n_steps == 0:
                epoch_loss_val = []
                test_sample_buffer.clear()
                test_buffer_count = 0

                model.eval()
                with torch.no_grad():
                    for val_step, batch_val in enumerate(test_dl):
                        if val_step >= config.max_val_steps:
                            break

                        # Buffer test samples up to 16
                        if test_buffer_count < TARGET_SAMPLE_COUNT:
                            test_sample_buffer.append(batch_val.detach().cpu())
                            test_buffer_count += batch_val.shape[0]

                        batch_val = batch_val.to(DEVICE)
                        model.to(DEVICE)

                        recon_images, mu, logvar = model(batch_val)
                        kl_beta = min(1.0, step / (config.max_steps * 0.1)) * config.kl_annealing
                        loss = loss_fn(recon_images, batch_val, mu, logvar, kl_beta=kl_beta)

                        epoch_loss_val.append(loss)

                        print(f"\rValidating. Items: {val_step * config.mini_batch_size}. Loss: {mean_loss(epoch_loss_val)['loss']:.5f}", end="")

                    print(f"\rStep: {step}. Running Loss: {mean_loss(epoch_loss_train)['loss']:.5f} Val Loss: {mean_loss(epoch_loss_val)['loss']:.5f}")
                    sample_model(-1, train_sample_buffer, test_sample_buffer, target_count=TARGET_SAMPLE_COUNT)

                # Log the metrics if enabled
                if config.mlflow.enabled:
                    mlflow.log_metrics({
                        **mean_loss(epoch_loss_train),
                        **{f"val_{key}": value for key, value in mean_loss(epoch_loss_val).items()}
                    }, step=step + 1)

                # Save model if validation loss improves
                current_step_loss = mean_loss(epoch_loss_val)['loss']
                if current_step_loss < best_val_loss:
                    best_val_loss = current_step_loss

                    model_path = f"{config.checkpoint_name}.pt"
                    torch.save(model.state_dict(), model_path)
                    export_onnx(model, "model.onnx")
