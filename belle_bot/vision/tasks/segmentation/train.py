from belle_bot.vision.tasks.segmentation.loss import OptimizedVaeLoss
from belle_bot.vision.tasks.utils.models import VisualCortexStepArgs

loss_fn = OptimizedVaeLoss()


def perform_step(args: VisualCortexStepArgs):
    batch_x, batch_y = args.batch

    loss_fn.to(args.device)
    batch_x = batch_x.to(args.device)
    batch_y = batch_y.to(args.device)

    pred_y = args.model(batch_x)

    return loss_fn(pred_y, batch_y)
