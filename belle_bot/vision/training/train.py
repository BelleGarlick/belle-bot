import math
import torch
from collections import deque
from contextlib import nullcontext

import mlflow
import numpy as np

from belle_bot.utils.cli import clpy
from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
import belle_bot.vision.tasks as tasks
from belle_bot.vision.tasks.utils.models import VisualCortexStepArgs
from belle_bot.vision.training.renderer import render_training_progress

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

config = clpy.parse_cli_args(VisionEncoderTrainingConfig())


tasks = {
    "encoder": tasks.encoder.get_task_definition(config, DEVICE),
    "decoder": tasks.decoder.get_task_definition(config, DEVICE),
    "segmentation": tasks.segmentation.get_task_definition(config, DEVICE),
    "detection": tasks.bounding_boxes.get_task_definition(config, DEVICE),
    "language": tasks.language.get_task_definition(config, DEVICE),
}


all_models = [x.model for x in tasks.values()]
active_tasks = ["decoder", "segmentation", "detection", "language"]

print("active_tasks:", active_tasks)

# Create joint optimiser for all model's parameters
model_params = []
for task in all_models:
    model_params.extend(task.parameters())
optimizer = torch.optim.AdamW(model_params, lr=config.learning_rate)


def mean_loss(data):
    if not data:
        return {"loss": 0}

    def get_val(val):
        return val.item() if isinstance(val, torch.Tensor) else val
    return {
        "loss": np.mean([get_val(x['loss']) for x in data]),
        # "recon_loss": np.mean([x['recon_loss'].item() for x in data]),
        # "rgb_loss": np.mean([x['rgb_loss'].item() for x in data]),
        # "depth_loss": np.mean([x['depth_loss'].item() for x in data]),
        # "perc_loss": np.mean([x['perc_loss'].item() for x in data]),
        # "kl_loss": np.mean([x['kl_loss'].item() for x in data])
    }


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
            g['lr'] = config.learning_rate * scale
        return config.learning_rate * scale

    with mlflow_run:
        if config.mlflow.enabled:
            mlflow.log_params(clpy.to_dict(config))

        epoch_loss_train = {key: deque(maxlen=2000) for key in list(tasks)}

        for step in range(config.max_steps):
            for model in all_models: model.train()

            active_task = active_tasks[step % len(active_tasks)]
            current_task = tasks[active_task]
            batch = next(current_task.train_set)

            percentage_complete = step / config.max_steps
            percentage_remaining = 1 - percentage_complete

            # scale the learning rate through training
            lr = update_learning_rate(percentage_remaining)

            optimizer.zero_grad()

            loss = current_task.perform_step(
                VisualCortexStepArgs(
                    config=config,
                    batch=batch,
                    vision_encoder=tasks['encoder'].model,
                    model=current_task.model,
                    percentage_complete=percentage_complete,
                    device=DEVICE,
                )
            )

            # Train the model
            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_train[active_task].append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss.items()})

            print(f"\rStep: {step}. Loss "
                  f"decoder: {mean_loss(epoch_loss_train["decoder"])['loss']:.5f} "
                  f"segmentation: {mean_loss(epoch_loss_train["segmentation"])['loss']:.5f} "
                  f"detection: {mean_loss(epoch_loss_train["detection"])['loss']:.5f} ",
                  f"language: {mean_loss(epoch_loss_train["language"])['loss']:.5f} ",
                end=""
            )

            # Validation
            if (step + 1) % config.eval_every_n_steps == 0:
                epoch_loss_val = {key: [] for key in list(tasks)}

                model.eval()
                with torch.no_grad():
                    for val_step in range(config.max_val_steps):
                        current_task = tasks[active_tasks[val_step % len(active_tasks)]]
                        batch = next(current_task.test_set)

                        val_loss = current_task.perform_step(
                            VisualCortexStepArgs(
                                config=config,
                                batch=batch,
                                vision_encoder=tasks['encoder'].model,
                                model=current_task.model,
                                percentage_complete=percentage_complete,
                                device=DEVICE,
                            )
                        )

                        epoch_loss_val[active_task].append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in val_loss.items()})

                        print(f"\rValidating. Items: {val_step * config.mini_batch_size}", end="")

                    print(f"\rStep: {step}. Loss "
                          f"decoder: {mean_loss(epoch_loss_train["decoder"])['loss']:.5f} "
                          f"segmentation: {mean_loss(epoch_loss_train["segmentation"])['loss']:.5f} "
                          f"detection: {mean_loss(epoch_loss_train["detection"])['loss']:.5f} ",
                          f"language: {mean_loss(epoch_loss_train["language"])['loss']:.5f} ",
                          f"Val Loss: ",
                          f"decoder: {mean_loss(epoch_loss_val["decoder"])['loss']:.5f} "
                          f"segmentation: {mean_loss(epoch_loss_val["segmentation"])['loss']:.5f} "
                          f"detection: {mean_loss(epoch_loss_val["detection"])['loss']:.5f} ",
                          f"language: {mean_loss(epoch_loss_val["language"])['loss']:.5f} ",
                    )
                    render_training_progress(
                        -1,
                        tasks=tasks,
                        test_dl=tasks["detection"].test_set,
                        device=DEVICE,
                    )

                # Save model if validation loss improves
                # todo this wont work if you're not trainnig the decoder
                #  need a way to workout how to do it. since this wont work for detection
                current_step_loss = mean_loss(epoch_loss_val['decoder'])['loss']
                if current_step_loss < best_val_loss:
                    best_val_loss = current_step_loss

                    for task in tasks:
                        torch.save(tasks[task].model.state_dict(), f"visual_cortex_v2b1_{task}.pt")

                # Log the metrics if enabled
                if config.mlflow.enabled:
                    mlflow.log_metrics({
                        **mean_loss(epoch_loss_train),
                        **{f"val_{key}": value for key, value in mean_loss(epoch_loss_val).items()}
                    }, step=step + 1)

            # Render images if the step % is reached
            if (step + 1) % config.render_image_every_n_steps == 0:
                render_training_progress(
                    step,
                    tasks=tasks,
                    test_dl=tasks["detection"].test_set,
                    device=DEVICE,
                )