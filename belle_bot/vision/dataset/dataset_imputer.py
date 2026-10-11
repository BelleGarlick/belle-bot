import io

import webdataset as wds
import numpy as np
from PIL import Image

from belle_bot.utils.cli import clpy
from belle_bot.vision.config.vision_encoder_dataset_creation_config import VisionEncoderDatasetCreationConfig
from ultralytics import YOLO


YOLO_FAMILY = "yolo26l"
DETECTION_YOLO = YOLO(f"{YOLO_FAMILY}.pt")
SEGMENTATION_YOLO = YOLO(f"{YOLO_FAMILY}-sem.pt")


def impute_dataset(dataset_path, max_items=10_000):
    dataset = wds.WebDataset(dataset_path)

    print(f"Generating embeddings for {dataset_path}...")
    with (
        wds.ShardWriter(str("distillation-test-%06d.tar"), maxcount=config.max_partition_size) as writer
    ):
        for i, item in enumerate(dataset):
            print(f"\r{i}/{max_items}", end="")
            if i > max_items:
                break

            main_image_bytes = io.BytesIO(item['rgb'])
            raw_main_image = np.array(Image.open(main_image_bytes))

            soft_labels = detect_objects(raw_main_image)
            seg_mask = compute_seg_mask(raw_main_image)

            writer.write({
                **item,
                "bounding_boxes": soft_labels.tobytes(),
                "segmentation_mask": seg_mask.tobytes(),
            })


def detect_objects(raw_rgb_image):
    """Runs the inference and processes the results."""
    response = DETECTION_YOLO.predict(
        raw_rgb_image,
        conf=0.000,
        iou=1.0,
        max_det=300,
        verbose=False
    )[0]
    boxes = response.boxes.data.numpy()

    # rescale the boxes so it's relative to the input image size
    return np.hstack((
        boxes[:, 0:1] / response.orig_shape[1],
        boxes[:, 1:2] / response.orig_shape[0],
        boxes[:, 2:3] / response.orig_shape[1],
        boxes[:, 3:4] / response.orig_shape[0],
        boxes[:, 4:]
    ))


def compute_seg_mask(raw_rgb_image):
    response = SEGMENTATION_YOLO(raw_rgb_image, verbose=False)[0]
    return response.semantic_mask.data.numpy()


if __name__ == "__main__":
    # Ensure config is parsed inside the main entry point
    config = clpy.parse_cli_args(VisionEncoderDatasetCreationConfig())
    clpy.print_values(config)

    # dataset_pattern = str(output_dir / "train-{000000..000001}.tar")

    file_path = "/Users/belle/Developer/belle-bot/belle_bot/vision/encoder/dataset/test-000000.tar"

    impute_dataset(file_path)
    print("Done!")
999999