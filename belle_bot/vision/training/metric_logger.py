from collections import deque

import numpy as np
import torch


class MetricLogger:
    def __init__(self, deque_length):
        self.deque_length = deque_length

        self.metrics = {}

    def __setitem__(self, key, value):
        self.metrics[key].append(value)

    def log(self, items: dict[str, float | torch.Tensor]):
        for key, value in items.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.deque_length)

            self.metrics[key].append(value)

    def write_training_metrics(self, step):
        print(f"\rStep: {step}. Loss: {np.mean(self.metrics['loss']):.5f} "
              f"decoder: {np.mean(self.metrics['decoder_loss']):.5f} "
              f"segmentation: {np.mean(self.metrics['segmentation_loss']):.5f} "
              f"bounding_boxes: {np.mean(self.metrics['bounding_boxes_loss']):.5f} ",
              # f"language: {np.mean(self.metrics['language_loss']):.5f} ",
          end=""
          )

    def write_validation_metrics(self, step):
        print(f"\rStep: {step}. Loss: {np.mean(self.metrics['loss']):.5f} "
              f"decoder: {np.mean(self.metrics['decoder_loss']):.5f} "
              f"segmentation: {np.mean(self.metrics['segmentation_loss']):.5f} "
              f"bounding_boxes: {np.mean(self.metrics['bounding_boxes_loss']):.5f} ",
              # f"language: {np.mean(self.metrics['language_loss']):.5f} ",
              f"Val Loss: {np.mean(self.metrics['val_loss']):.5f} ",
              f"decoder: {np.mean(self.metrics['val_decoder_loss']):.5f} "
              f"segmentation: {np.mean(self.metrics['val_segmentation_loss']):.5f} "
              f"bounding_boxes: {np.mean(self.metrics['val_bounding_boxes_loss']):.5f} ",
              # f"language: {np.mean(self.metrics['val_language_loss']):.5f} ",
          )
