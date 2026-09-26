import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import webdataset as wds
from sklearn.metrics.pairwise import cosine_similarity

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_dataset_creation_config import VisionEncoderDatasetCreationConfig
from belle_bot.vision.encoder.training.data_loader import custom_decoder, merge

config = clpy.parse_cli_args(VisionEncoderDatasetCreationConfig())
clpy.print_values(config)


def load_encoder(model_path: str):
    providers = ['CPUExecutionProvider']
    if torch.backends.mps.is_available():
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    elif torch.cuda.is_available():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    return ort.InferenceSession(model_path, providers=providers)


def get_embeddings(encoder, dataset_path):
    dataset = wds.WebDataset(dataset_path)
        # .map(custom_decoder)
        # .map(merge)

    embeddings = []
    keys = []

    print(f"Generating embeddings for {dataset_path}...")
    for i, item in enumerate(dataset):
        input_item = merge(custom_decoder(item))

        # item is HWC [0, 1]
        input_tensor = np.transpose(input_item, (2, 0, 1)).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        ort_inputs = {encoder.get_inputs()[0].name: input_tensor}
        encoded = encoder.run(None, ort_inputs)[0]

        # Flatten the embedding if it's not already
        embeddings.append(encoded.flatten())
        keys.append(item['frame_id'].decode("utf8")) # WebDataset doesn't always have a clear key in the map result if we don't return it

        if i % 100 == 0:
            print(f"\rProcessed {i} items", end="", flush=True)

    print(f"\nFinished generating {len(embeddings)} embeddings.")
    return np.array(embeddings), keys


if __name__ == "__main__":
    # step 1: load up the datasets given in the config path
    output_dir = Path(config.output_dir)
    # We'll look for train shards in the output_dir
    dataset_pattern = str(output_dir / "train-{000000..000002}.tar")

    # step 3: load up the onnx encoder in this dir and create an embedding for all items
    encoder_path = os.path.join(os.path.dirname(__file__), "../model_exporter/model-729_2_encoder.onnx")
    if not os.path.exists(encoder_path):
        # Fallback to absolute path from run.py if needed, but let's try relative first
        encoder_path = "/Users/belle/Developer/belle-bot/belle_bot/vision/encoder/model_exporter/model-729_2_encoder.onnx"

    print(f"Loading encoder from {encoder_path}")
    encoder = load_encoder(encoder_path)

    embeddings, keys = get_embeddings(encoder, dataset_pattern)

    if len(embeddings) == 0:
        raise Exception("No items found in dataset.")

    # step 4: count the number of items with either near cosine similarity or euclidean distance
    print("Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)

    # Threshold for "near" similarity
    threshold = 0.995

    # Step 5: report the items with the highest clusters
    # A simple way to find clusters: for each item, count how many other items are "near" it
    near_counts = (sim_matrix > threshold).sum(axis=1) # subtract 1 to exclude self

    # Save the frequency map
    frequency_map = {key: int(value) for key, value in zip(keys, near_counts)}
    with open("frequency_map.json", "w+") as f:
        f.write(json.dumps(frequency_map, indent=2))
