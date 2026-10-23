import math
import torch
torch.autograd.set_detect_anomaly(True)
from collections import deque
from contextlib import nullcontext

import mlflow
import numpy as np

from belle_bot.mapping.positioning.config.positioning_config import MlFlowConfig
from belle_bot.utils.cli import clpy
from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
import belle_bot.vision.tasks as tasks
from belle_bot.vision.tasks.utils.models import VisualCortexStepArgs
from belle_bot.vision.training import dataloader
from belle_bot.vision.training.renderer import render_training_progress

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# todo clean up the old data loaders
# todo allow for language training
# todo clean up the training perform steps

# Dataset setup
# train_path = str("/run/media/belle/Houston/datasets/vision-encoder/v1/train-{000000..000036}.tar")
# test_path = str("/run/media/belle/Houston/datasets/vision-encoder/v1/test-{000000..000007}.tar")

config = clpy.parse_cli_args(VisionEncoderTrainingConfig(
    training_path="/Users/belle/Developer/belle-bot/belle_bot/vision/training/distillation-test-{000000..000001}.tar",
    testing_path="/Users/belle/Developer/belle-bot/belle_bot/vision/training/distillation-test-{000000..000001}.tar",
    mlflow=MlFlowConfig(
        enabled=False
    ),
    eval_every_n_steps=500,
    max_val_steps=100,
    mini_batch_size=4
))

tasks = {
    "encoder": tasks.encoder.get_task_definition(config, DEVICE),
    "decoder": tasks.decoder.get_task_definition(config, DEVICE),
    "segmentation": tasks.segmentation.get_task_definition(config, DEVICE),
    "bounding_boxes": tasks.bounding_boxes.get_task_definition(config, DEVICE),
    # "language": tasks.language.get_task_definition(config, DEVICE),
}


all_models = [x.model for x in tasks.values()]
active_tasks = ["decoder", "segmentation", "bounding_boxes"]
print("active_tasks:", active_tasks)

training_set, testing_set = dataloader.load_dataset(config, DEVICE)

# Create joint optimiser for all model's parameters
encoder_params = tasks['encoder'].model.parameters()
decoder_params = []
for task_name, task_def in tasks.items():
    if task_name != 'encoder':
        decoder_params.extend(task_def.model.parameters())

optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': config.learning_rate * config.encoder_learning_rate_factor, 'base_lr': config.learning_rate * config.encoder_learning_rate_factor},
    {'params': decoder_params, 'lr': config.learning_rate, 'base_lr': config.learning_rate}
])
model_params = list(tasks['encoder'].model.parameters()) + decoder_params


if __name__ == "__main__":
    best_val_loss = math.inf

    if config.mlflow.enabled:
        mlflow.set_tracking_uri(config.mlflow.endpoint)
        mlflow.set_experiment("vision-encoder")
        mlflow_run = mlflow.start_run(run_name=str("optimised-vae"))
    else:
        mlflow_run = nullcontext()

    def update_learning_rate(percentage_remaining):
        scale = math.pow(percentage_remaining, config.learning_rate_gamma)
        for g in optimizer.param_groups:
            g['lr'] = g['base_lr'] * scale

        return config.learning_rate * scale

    with mlflow_run:
        if config.mlflow.enabled:
            mlflow.log_params(clpy.to_dict(config))

        epoch_loss_train = {}

        for step in range(config.max_steps):
            for model in all_models: model.train()

            percentage_complete = step / config.max_steps
            percentage_remaining = 1 - percentage_complete

            # scale the learning rate through training
            lr = update_learning_rate(percentage_remaining)

            optimizer.zero_grad()

            loss = torch.zeros(1).to(DEVICE)
            step_metrics = {}

            # todo run encoder once here
            # todo have way for frozen stuff and only fine tuning a head

            batch = next(training_set)

            latent_encoding, mu, logvar = tasks['encoder'].model(batch['input'].to(DEVICE))

            for task in active_tasks:
                current_task = tasks[task]

                task_loss = current_task.perform_step(
                    VisualCortexStepArgs(
                        config=config,
                        batch=(latent_encoding, batch[task]),
                        model=current_task.model,
                        percentage_complete=percentage_complete,
                        device=DEVICE,
                    )
                )

                # Add all metrics to the step_results
                loss += task_loss['loss'] / len(active_tasks)
                for key, value in task_loss.items():
                    if isinstance(value, torch.Tensor): value = value.item()
                    step_metrics[task + "_" + key] = value

            # KL Annealing: start at 0, increase to 1
            logvar_clamped = logvar.clamp(-20, 20)
            kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
            kl_beta = percentage_complete * config.kl_annealing
            kl_loss = kl_loss * kl_beta
            step_metrics['kl_loss'] = kl_loss.item()
            loss += kl_loss

            # Train the model
            loss.backward()
            step_metrics['loss'] = loss.item()
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=0.5)
            optimizer.step()

            for key, value in step_metrics.items():
                if key not in epoch_loss_train: epoch_loss_train[key] = deque(maxlen=2000)
                epoch_loss_train[key].append(value)

            print(f"\rStep: {step}. Loss: {np.mean(epoch_loss_train['loss']):.5f} "
                  f"decoder: {np.mean(epoch_loss_train['decoder_loss']):.5f} "
                  f"segmentation: {np.mean(epoch_loss_train['segmentation_loss']):.5f} "
                  f"bounding_boxes: {np.mean(epoch_loss_train['bounding_boxes_loss']):.5f} ",
                  # f"language: {np.mean(epoch_loss_train['language_loss']):.5f} ",
                end=""
            )

            # Validation
            if (step + 1) % config.eval_every_n_steps == 0:
                for model in all_models: model.eval()
                epoch_loss_val = {}

                with torch.no_grad():
                    for val_step in range(config.max_val_steps):
                        val_loss = torch.zeros(1).to(DEVICE)
                        val_step_metrics = {}

                        batch = next(testing_set)
                        for task in active_tasks:
                            current_task = tasks[task]

                            latent_encoding, mu, logvar = tasks['encoder'].model(batch['input'].to(DEVICE))
                            task_loss = current_task.perform_step(
                                VisualCortexStepArgs(
                                    config=config,
                                    batch=(mu, batch[task]),
                                    model=current_task.model,
                                    percentage_complete=percentage_complete,
                                    device=DEVICE,
                                )
                            )

                            # Add all metrics to the step_results
                            val_loss += task_loss['loss'] / len(active_tasks)
                            for key, value in task_loss.items():
                                if isinstance(value, torch.Tensor): value = value.item()
                                val_step_metrics['val_' + task + "_" + key] = value

                        val_step_metrics['val_loss'] = val_loss.item()

                        for key, value in val_step_metrics.items():
                            if key not in epoch_loss_val: epoch_loss_val[key] = deque(maxlen=2000)
                            epoch_loss_val[key].append(value)

                        print(f"\rValidating. Items: {val_step * config.mini_batch_size}", end="")

                    print(f"\rStep: {step+1}. Loss: {np.mean(epoch_loss_train['loss']):.5f} "
                          f"decoder: {np.mean(epoch_loss_train['decoder_loss']):.5f} "
                          f"segmentation: {np.mean(epoch_loss_train['segmentation_loss']):.5f} "
                          f"bounding_boxes: {np.mean(epoch_loss_train['bounding_boxes_loss']):.5f} ",
                          # f"language: {np.mean(epoch_loss_train['language_loss']):.5f} ",
                          f"Val Loss: {np.mean(val_step_metrics['val_loss']):.5f} ",
                          f"decoder: {np.mean(val_step_metrics['val_decoder_loss']):.5f} "
                          f"segmentation: {np.mean(val_step_metrics['val_segmentation_loss']):.5f} "
                          f"bounding_boxes: {np.mean(val_step_metrics['val_bounding_boxes_loss']):.5f} ",
                          # f"language: {np.mean(val_step_metrics['val_language_loss']):.5f} ",
                      )
                    render_training_progress(
                        -1,
                        tasks=tasks,
                        test_dl=testing_set,
                        device=DEVICE,
                    )

                # Save model if validation loss improves
                # todo this wont work if you're not trainnig the decoder
                #  need a way to workout how to do it. since this wont work for bounding_boxes
                current_step_loss = np.mean(val_step_metrics['val_loss'])
                if current_step_loss < best_val_loss:
                    best_val_loss = current_step_loss

                    for task in tasks:
                        torch.save(tasks[task].model.state_dict(), f"visual_cortex_v2b2_{task}.pt")

                # Log the metrics if enabled
                if config.mlflow.enabled:
                    mlflow.log_metrics({
                        **epoch_loss_train,
                        **val_step_metrics,
                    }, step=step + 1)

            # Render images if the step % is reached
            if (step + 1) % config.render_image_every_n_steps == 0:
                render_training_progress(
                    step,
                    tasks=tasks,
                    test_dl=testing_set,
                    device=DEVICE,
                )
