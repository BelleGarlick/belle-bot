import math
from collections import deque

import matplotlib.pyplot as plt
import mlflow
import numpy as np

import torch

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.encoder.training.data_loader import load_dataset
from belle_bot.vision.encoder.training.loss import OptimizedVaeLoss
from belle_bot.vision.encoder.training.ml_model import VAE, OptimizedVAE

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

config = clpy.parse_cli_args(VisionEncoderTrainingConfig())

model = OptimizedVAE(latent_dim=config.model.embedding_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
loss_fn = OptimizedVaeLoss().to(DEVICE)

train_dataset, test_dataset = load_dataset(config, DEVICE)

def sample_model(step, train_dl, test_dl):
    model.eval()
    with torch.no_grad():
        # Get one batch from train and test
        for name, dl in [("train", train_dl), ("test", test_dl)]:
            (batch,) = next(iter(dl))
            x = batch
            
            recon, _, _ = model(x)
            
            y = recon.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()

            image_pairs = [
                np.vstack((
                    input_image.transpose((1, 2, 0)),
                    output_image.transpose((1, 2, 0))
                ))
                for input_image, output_image in zip(x_np, y)
            ]
            row = np.hstack(image_pairs[:4]) # Show 4 pairs

            plt.figure(figsize=(12, 6))
            plt.title(f"Reconstructions ({name})")
            plt.imshow(np.clip(row, 0, 1))
            plt.savefig(f"training_plot_{(step+1)}_{name}.png")
            plt.close()


if __name__ == "__main__":
    train_dl = iter(train_dataset)
    test_dl = iter(test_dataset)

    mlflow.set_tracking_uri(config.mlflow.endpoint)
    mlflow.set_experiment("vision-encoder")

    with mlflow.start_run(run_name=str("optimised-vae")):
    # with mlflow.start_run(run_name=str(uuid.uuid4())):
        mlflow.log_params(clpy.to_dict(config))
        # mlflow.set_tag("experiment", EXPERIMENT_TAG)

        epoch_loss_train = deque(maxlen=1000)
        for step, (batch, ) in enumerate(train_dl):
            if step >= config.max_steps:
                break

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
            loss = loss_fn(recon_images, batch, mu, logvar, kl_beta=percentage_remaining * config.kl_annealing)['loss']

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
                # We use a fresh iterator for validation
                temp_test_dl = iter(test_dataset)
                with torch.no_grad():
                    for _ in range(20): # Validate on 20 batches
                        try:
                            (batch,) = next(temp_test_dl)
                        except StopIteration:
                            break

                        recon_images, mu, logvar = model(batch)
                        loss = loss_fn(recon_images, batch, mu, logvar, kl_beta=percentage_remaining * config.kl_annealing)['loss']

                        epoch_loss_val.append(loss.item())
                        val_samples_collected += batch.shape[0]
                        print(f"\rValidating. Items: {val_samples_collected}. Loss: {np.mean(epoch_loss_val):.5f}", end="")

                    print(f"\rStep: {step}. Running Loss: {np.mean(epoch_loss_train):.5f} Val Loss: {np.mean(epoch_loss_val):.5f}")

                    mlflow.log_metrics({
                        "loss": np.mean(epoch_loss_train),
                        "val_loss": np.mean(epoch_loss_val),
                    }, step=step + 1)

                    # Print some example outputs
                    sample_model(step, train_dataset, test_dataset)

            if (step + 1) % config.checkpoint_every_n_steps == 0:
                model_path = f"model-{step + 1}.pt"
                torch.save(model.state_dict(), model_path)
