import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import webdataset as wds

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_dataset_creation_config import VisionEncoderDatasetCreationConfig
from belle_bot.vision.encoder.training.data_loader import custom_decoder, merge


def load_encoder(model_path: str):
    providers = ['CPUExecutionProvider']
    if torch.backends.mps.is_available():
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    elif torch.cuda.is_available():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    return ort.InferenceSession(model_path, providers=providers)


def get_embeddings(encoder, dataset_path):
    dataset = wds.WebDataset(dataset_path)

    embeddings = []
    keys = []

    print(f"Generating embeddings for {dataset_path}...")
    for i, item in enumerate(dataset):
        input_item = merge(custom_decoder(item))

        # Prepare input tensor
        input_tensor = np.transpose(input_item, (2, 0, 1)).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        ort_inputs = {encoder.get_inputs()[0].name: input_tensor}
        encoded = encoder.run(None, ort_inputs)[0].flatten()

        # L2-normalize immediately so dot product equals cosine similarity
        norm = np.linalg.norm(encoded)
        if norm > 0:
            encoded /= norm

        embeddings.append(encoded.astype(np.float32))
        keys.append(item['frame_id'].decode("utf8"))

        if i % 100 == 0:
            print(f"\rProcessed {i} items", end="", flush=True)

    print(f"\nFinished generating {len(embeddings)} embeddings.")
    return np.array(embeddings, dtype=np.float32), keys


def compute_near_counts_batched(embeddings, threshold=0.995, batch_size=1000):
    num_samples = embeddings.shape[0]
    near_counts = np.zeros(num_samples, dtype=np.int32)

    print(f"Computing near counts in batches of {batch_size}...")
    for i in range(0, num_samples, batch_size):
        end_i = min(i + batch_size, num_samples)
        batch = embeddings[i:end_i]

        # Dot product of unit vectors = Cosine Similarity
        sim_batch = np.dot(batch, embeddings.T)

        near_counts[i:end_i] = (sim_batch > threshold).sum(axis=1)
        print(f"\rProcessed chunk {i}/{num_samples}", end="", flush=True)

    print("\nFinished similarity computation.")
    return near_counts


if __name__ == "__main__":
    # Ensure config is parsed inside the main entry point
    config = clpy.parse_cli_args(VisionEncoderDatasetCreationConfig())
    clpy.print_values(config)

    output_dir = Path("/run/media/belle/Houston/datasets/vision-encoder/v0")
    dataset_pattern = str(output_dir / "train-{000000..000004}.tar")

    encoder_path = "/home/belle/Developer/belle-bot/belle_bot/vision/encoder/dataset/model-1024_2_encoder.onnx"
    print(f"Loading encoder from {encoder_path}")
    encoder = load_encoder(encoder_path)

    embeddings, keys = get_embeddings(encoder, dataset_pattern)

    if len(embeddings) == 0:
        raise Exception("No items found in dataset.")

    threshold = 0.995
    near_counts = compute_near_counts_batched(embeddings, threshold=threshold, batch_size=1000)

    frequency_map = {key: int(value) for key, value in zip(keys, near_counts)}
    with open("frequency_map.json", "w") as f:
        json.dump(frequency_map, f, indent=2)
