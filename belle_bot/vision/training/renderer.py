import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from belle_bot.vision.tasks.bounding_boxes.utils import perform_nms, render_boxes
from belle_bot.vision.tasks.utils.models import TaskDefinition


def render_training_progress(
        step,
        tasks: dict[str, TaskDefinition],
        test_dl,
        device: torch.device,
        target_count=16,
):
    """
    Renders up to `target_count` samples in a dynamic grid layout.

    `train_samples` and `test_samples` are lists/deques of batch tensors.
    """
    for task in tasks:
        tasks[task].model.eval()

    cm = plt.get_cmap('rainbow')
    def render_depth(d):
        d = d / (np.max(d) + 1e-8)
        return cm(d)[:, :, :3]

    with torch.no_grad():
        image_pairs = []
        for b_item in test_dl:
            batch = b_item[0].to(device)

            x_np = batch.detach().cpu().numpy()

            batch_latent = tasks['encoder'].model(batch)[0]
            batch_detection_y = tasks['detection'].model(batch_latent)
            batch_decoder_y = tasks['decoder'].model(batch_latent)
            batch_segmentation_y = tasks['segmentation'].model(batch_latent)

            batch_detection_y = batch_detection_y.detach().cpu().numpy()
            batch_decoder_y = batch_decoder_y.detach().cpu().numpy()

            batch_segmentation_y = torch.argmax(F.softmax(batch_segmentation_y, dim=1), dim=1)
            batch_segmentation_y = batch_segmentation_y.detach().cpu().numpy()

            batch_detection_y = batch_detection_y.reshape((batch.shape[0], 300, 85))

            batch_detection_y = np.concatenate((
                batch_detection_y[:, :, :4] * 448,
                batch_detection_y[:, :, 4:5],
                np.expand_dims(np.argmax(batch_detection_y[:, :, 5:], axis=2), axis=2)
            ), axis=2)

            for i in range(batch_decoder_y.shape[0]):
                input_rgb = (x_np[i, :3, :, :] * 255).transpose((1, 2, 0)).astype(np.uint8)
                output_rgb = (batch_decoder_y[i, :3, :, :] * 255).transpose((1, 2, 0)).astype(np.uint8)
                output_depth = (render_depth(batch_decoder_y[i, 3, :, :]) * 255).astype(np.uint8)
                output_segmentation = (render_depth(batch_segmentation_y[i] / 80) * 255).astype(np.uint8)

                boxes = perform_nms(torch.tensor(batch_detection_y[i:i+1]))[0]
                input_rgb = render_boxes(input_rgb, boxes.detach().cpu().numpy())

                # Create the image grid for a single item (Top: Input RGB+Depth, Bottom: Output RGB+Depth)
                pair = np.vstack((
                    np.hstack((input_rgb, output_rgb)),
                    np.hstack((output_depth, output_segmentation))
                ))
                image_pairs.append(pair)

            if len(image_pairs) >= target_count:
                break

        # Arrange into rows dynamically (up to 4 items per row)
        items_per_row = 4
        rows = []
        for r in range(0, target_count, items_per_row):
            row_items = image_pairs[r:r + items_per_row]
            if row_items:
                rows.append(np.hstack(row_items))

        grid = np.vstack(rows)

        plt.figure(figsize=(32, 8 * len(rows)))
        plt.title(f"Reconstructions - Left: RGB, Right: Depth | Top: Input, Bottom: Output")
        plt.imshow(np.clip(grid, 0, 255))
        plt.savefig(f"visualisation_{step}.png")
        #plt.show()
        plt.close()

