import base64
import json
from typing import Literal

import random
import multiprocessing as mp
from functools import partial

import numpy as np
import opensimplex
import webdataset as wds
from tqdm import tqdm

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_dataset_creation_config import VisionEncoderDatasetCreationConfig
from houston.client.py import replays

# todo upload the dataset to houston once done

config = clpy.parse_cli_args(VisionEncoderDatasetCreationConfig())
clpy.print_values(config)


def get_replay_ids(subset: Literal["train", "test"] | None):
    filter = ["dataset/vision/encoder"]
    if subset:
        filter += [subset]

    replay_ids = replays.query_replays(
        config.houston,
        page=0,
        tags=filter
    )['replays']

    return sorted([x["replay_id"] for x in replay_ids])


def generate_single_mask(idx, size):
    gen = opensimplex.OpenSimplex(seed=idx)
    x_idxs = np.arange(size)
    y_idxs = np.arange(size)

    low_freq = gen.noise2array(x=x_idxs / 200,
                               y=y_idxs / 200) * 0.8
    high_freq = gen.noise2array(x=size + x_idxs / 30,
                                y=size + y_idxs / 30) * 0.2

    total_noise = ((low_freq + high_freq) + 1) / 2
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[total_noise > 0.65] = 1
    return mask


def create_masks(config: VisionEncoderDatasetCreationConfig):
    if not config.mask:
        return None

    print(f"Generating {config.mask.count} masks using {mp.cpu_count()} processes...")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        func = partial(generate_single_mask, size=config.mask.size)
        masks = list(tqdm(pool.imap(func, range(config.mask.count)), total=config.mask.count, desc="Generating masks"))

    print("Finished generating masks")

    return np.array(masks, dtype=np.uint8)


def create_dataset(shuffle=True):
    replay_ids = get_replay_ids(config.subset)

    masks = create_masks(config)

    # todo at somepoint this may cause memory to grow too large.
    #  at which point we will need to create numerous dataset files
    #  and then in a second step shuffle items from them around and keep mixing them all up
    
    all_items = []

    frequency_map = {}
    if config.frequency_map:
        with open(config.frequency_map) as f:
            frequency_map = json.load(f)

    skipped_frames = 0
    for replay_idx, replay_id in enumerate(replay_ids):
        replay_file = replays.get_replay_file(config.houston, replay_id)
        lines = replay_file.split("\n")

        frame_idx = 0
        for line in lines:
            print(f"\r{config.subset} Reading {replay_idx}/{len(replay_ids)} {replay_id}", end="")

            split_tokens = line.split(",")
            if len(split_tokens) < 3:
                continue

            stream = split_tokens[0]
            timestamp = float(split_tokens[1])
            data = json.loads(",".join(split_tokens[2:]))

            if stream == "sensors/camera":
                key = f"{replay_id}/{frame_idx}"

                likelihood = 1 / frequency_map.get(key, 1)
                if random.random() <= likelihood:
                    all_items.append({
                        "key": key,
                        "replay_id": replay_id,
                        "timestamp": timestamp,
                        "rgb": data['rgb'],
                        "depth": data['depth']
                    })
                else:
                    skipped_frames += 1

                frame_idx += 1

    print(f"\n{config.subset} Total items collected: {len(all_items)} (skipped {skipped_frames}). {'Shuffling...' if shuffle else ''}")
    if shuffle:
        random.shuffle(all_items)

    with (
        wds.ShardWriter(str(config.path), maxcount=config.max_partition_size) as writer
    ):
        for item_count, item in enumerate(tqdm(all_items, desc=f"{config.subset} Writing")):
            try:
                rgb_bytes = base64.b64decode(item['rgb'])
                depth_bytes = base64.b64decode(item['depth'])
            except Exception:
                rgb_bytes = item['rgb']
                depth_bytes = item['depth']

            data_to_write = {
                "__key__": f"{item_count}",
                "rgb": rgb_bytes,
                "depth": depth_bytes,
                "frame_id": item['key'],
                "json": {
                    "replay_id": item['replay_id'],
                    "timestamp": item['timestamp'],
                },
            }

            if masks is not None:
                mask_idx = random.randint(0, len(masks) - 1)
                mask = masks[mask_idx]
                
                # Randomly rotate the mask
                rotation_k = random.randint(0, 3)
                mask = np.rot90(mask, k=rotation_k)
                
                data_to_write["depth_mask"] = mask.tobytes()

            writer.write(data_to_write)
    print()


if __name__ == "__main__":
    create_dataset()
