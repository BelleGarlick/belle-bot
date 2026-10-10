import torch

from belle_bot.vision.tasks.bounding_boxes.loss import YOLODistillationLoss
from belle_bot.vision.tasks.utils.models import VisualCortexStepArgs

loss_fn = YOLODistillationLoss()


def perform_step(args: VisualCortexStepArgs):
    batch_x, batch_y = args.batch

    loss_fn.to(args.device)
    batch_x = batch_x.to(args.device)
    batch_y = batch_y.to(args.device)

    latent_encoding, mu, logvar = args.vision_encoder(batch_x)
    pred_y = args.model(latent_encoding)

    task_loss = loss_fn(pred_y, batch_y)

    # KL Annealing: start at 0, increase to 1
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_beta = args.percentage_complete * args.config.kl_annealing
    task_loss['kl_loss'] = kl_beta * kl_loss

    # Train the model
    task_loss['loss'] = task_loss['loss'] + task_loss['kl_loss']

    return task_loss
