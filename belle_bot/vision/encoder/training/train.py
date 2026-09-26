import math
from collections import deque
from contextlib import nullcontext

import matplotlib.pyplot as plt
import mlflow
import numpy as np

import torch

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.encoder.training.data_loader import load_dataset
from belle_bot.vision.encoder.training.loss import OptimizedVaeLoss
from belle_bot.vision.encoder.training.ml_model import VAE2_448

from torch.utils.data import DataLoader

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

config = clpy.parse_cli_args(VisionEncoderTrainingConfig())
config.houston.enabled = False

model = VAE2_448(latent_dim=config.model.embedding_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
loss_fn = OptimizedVaeLoss().to(DEVICE)

train_dataset, test_dataset = load_dataset(config)
pin_memory = DEVICE.type != 'mps'
train_loader = DataLoader(train_dataset, batch_size=None, num_workers=4, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=None, num_workers=2, pin_memory=pin_memory)

# todo
#  mask out 0 depth data items so the model has to learn how to infer them
#  report all training stats. also report the eval stats better
#  have a fully trained model and explore random generation
#  change the mid rendering to use the normalised value
#  have have a way to weight the items
#  possible have the encoder trained via a multi-task embedding
#  possibly drop input data and have hte model try and predict both
#  have an automated pipeline for tagging items if their error is high enough

def sample_model(step, train_batch, test_batch):
    model.eval()
    with torch.no_grad():
        # Get one batch from train and test
        for name, batch in [("train", train_batch), ("test", test_batch)]:
            recon, _, _ = model(batch)
            
            y = recon.detach().cpu().numpy()
            x_np = batch.detach().cpu().numpy()

            # y and x_np are shape (B, 4, 224, 224)
            # We want to display them. We can stack RGB and Depth vertically or side-by-side.
            # Let's show RGB and Depth as separate rows for both input and output.

            image_pairs = []
            cm = plt.get_cmap('rainbow')
            for i in range(min(4, x_np.shape[0])):
                input_rgb = x_np[i, :3, :, :].transpose((1, 2, 0))
                input_depth = cm(x_np[i, 3, :, :])[:, :, :3]
                output_rgb = y[i, :3, :, :].transpose((1, 2, 0))
                output_depth = cm(y[i, 3, :, :])[:, :, :3]

                # Normalize depth for visualization if needed, but it should be 0-1

                pair = np.vstack((
                    np.hstack((input_rgb, input_depth)),
                    np.hstack((output_rgb, output_depth))
                ))
                image_pairs.append(pair)
            
            row = np.hstack(image_pairs)

            plt.figure(figsize=(16, 8))
            plt.title(f"Reconstructions ({name}) - Left: RGB, Right: Depth | Top: Input, Bottom: Output")
            plt.imshow(np.clip(row, 0, 1))
            plt.savefig(f"training_plot_{(step+1)}_{name}_1.5_{config.model.embedding_size}_2.png")
            plt.close()


if __name__ == "__main__":
    train_dl = iter(train_loader)
    test_dl = iter(test_loader)

    if config.houston.enabled:
        mlflow.set_tracking_uri(config.mlflow.endpoint)
        mlflow.set_experiment("vision-encoder")
        mlflow_run = mlflow.start_run(run_name=str("optimised-vae"))
    else:
        mlflow_run = nullcontext()

    with mlflow_run:
        if config.houston.enabled:
            mlflow.log_params(clpy.to_dict(config))

        epoch_loss_train = deque(maxlen=1000)
        for step, batch in enumerate(train_dl):
            if step >= config.max_steps:
                break
            
            batch = batch.to(DEVICE, non_blocking=True)

            percentage_complete = step / config.max_steps
            percentage_remaining = 1 - percentage_complete

            # scale the learning rate through training
            scale = 1 - math.pow(percentage_remaining, config.learning_rate_gamma)
            for g in optimizer.param_groups:
                g['lr'] = config.learning_rate * scale

            model.train()
            optimizer.zero_grad()

            # Feed through model & compute loss
            recon_images, mu, logvar = model(batch)
            # KL Annealing: start at 0, increase to 1
            kl_beta = percentage_complete * config.kl_annealing
            loss = loss_fn(recon_images, batch, mu, logvar, kl_beta=kl_beta)['loss']

            # Train the model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_train.append(loss.item())
            print(f"\rStep: {step}. Loss: {np.mean(epoch_loss_train):.5f}", end="")

            if (step + 1) % config.eval_every_n_steps == 0:
                epoch_loss_val = []
                model.eval()

                # Validation
                val_samples_collected = 0
                # We use the existing iterator for validation to avoid re-initializing
                last_test_batch = None
                with torch.no_grad():
                    for _ in range(20): # Validate on 20 batches
                        try:
                            batch_val = next(test_dl)
                            batch_val = batch_val.to(DEVICE, non_blocking=True)
                            last_test_batch = batch_val
                        except StopIteration:
                            # Re-initialize if we hit the end (though with resampled=True it shouldn't)
                            test_dl = iter(test_loader)
                            try:
                                batch_val = next(test_dl)
                                batch_val = batch_val.to(DEVICE, non_blocking=True)
                                last_test_batch = batch_val
                            except StopIteration:
                                break

                        recon_images, mu, logvar = model(batch_val)
                        kl_beta = min(1.0, step / (config.max_steps * 0.1)) * config.kl_annealing
                        loss = loss_fn(recon_images, batch_val, mu, logvar, kl_beta=kl_beta)['loss']

                        epoch_loss_val.append(loss.item())
                        val_samples_collected += batch_val.shape[0]
                        print(f"\rValidating. Items: {val_samples_collected}. Loss: {np.mean(epoch_loss_val):.5f}", end="")

                    print(f"\rStep: {step}. Running Loss: {np.mean(epoch_loss_train):.5f} Val Loss: {np.mean(epoch_loss_val):.5f}")

                    if config.houston.enabled:
                        mlflow.log_metrics({
                            "loss": np.mean(epoch_loss_train),
                            "val_loss": np.mean(epoch_loss_val),
                        }, step=step + 1)

                    # Print some example outputs
                    if last_test_batch is not None:
                        # We use the current training batch and the last validation batch
                        sample_model(step, batch, last_test_batch)

            if (step + 1) % config.checkpoint_every_n_steps == 0:
                model_path = f"model-{step + 1}-1.5-{config.model.embedding_size}_2.pt"
                torch.save(model.state_dict(), model_path)
