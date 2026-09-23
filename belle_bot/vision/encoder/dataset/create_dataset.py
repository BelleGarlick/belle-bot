import base64
import io
import json
from pathlib import Path
from typing import Literal

import random

import webdataset as wds
from PIL import Image
from matplotlib import pyplot as plt

from belle_bot.utils.cli.clpy import parse_cli_args
from belle_bot.vision.encoder.config.vision_encoder_dataset_creation_config import VisionEncoderDatasetCreationConfig
from houston.client.py import replays

# todo upload the dataset to houston once done

config = parse_cli_args(VisionEncoderDatasetCreationConfig())


def get_replay_ids(subset: Literal["train", "eval"] | None):
    filter = []
    if subset:
        filter += [subset]

    replay_ids = replays.query_replays(
        config.houston,
        page=0,
        tags=filter
    )['replays']

    return sorted([x["replay_id"] for x in replay_ids])


def create_dataset(subset: Literal["train", "eval"], pattern: Path, shuffle=True):
    replay_ids = get_replay_ids(subset)

    # todo at somepoint this may cause memory to grow too large.
    #  at which point we will need to create numerous dataset files
    #  and then in a second step shuffle items from them around and keep mixing them all up
    
    all_items = []

    for replay_idx, replay_id in enumerate(replay_ids):
        replay_file = replays.get_replay_file(config.houston, replay_id)
        lines = replay_file.split("\n")

        for line in lines:
            print(f"\r{subset} Reading {replay_idx}/{len(replay_ids)} {replay_id}", end="")

            split_tokens = line.split(",")
            if len(split_tokens) < 3:
                continue

            stream = split_tokens[0]
            timestamp = float(split_tokens[1])
            data = json.loads(",".join(split_tokens[2:]))

            import numpy as np
            if stream == "sensors/camera":
                main_image = Image.open(io.BytesIO(base64.b64decode(data['rgb'])))

                depth_raw_buffer = np.frombuffer(base64.b64decode(data['depth_raw']), dtype=np.float16)
                depth_raw_buffer = np.resize(depth_raw_buffer, (320 // 2, 480 // 2, 1))

                _, color_buffer = cv2.imencode('.png', depth_raw_buffer)

                # todo further testing

                # 480/320
                all_items.append({
                    "replay_id": replay_id,
                    "timestamp": timestamp,
                    "rgb": data['rgb'],
                    "depth": data['depth']
                })

    
    print(f"\n{subset} Total items collected: {len(all_items)}. {'Shuffling...' if shuffle else ''}")
    if shuffle:
        random.shuffle(all_items)

    with (
        wds.ShardWriter(str(pattern), maxcount=config.max_partition_size) as writer
    ):
        for item_count, item in enumerate(all_items):
            print(f"\r{subset} Writing {item_count}/{len(all_items)}", end="")

            try:
                rgb_bytes = base64.b64decode(item['rgb'])
                depth_bytes = base64.b64decode(item['depth'])
            except Exception:
                rgb_bytes = item['rgb']
                depth_bytes = item['depth']

            writer.write({
                "__key__": f"{item_count}",
                "rgb": rgb_bytes,
                "depth": depth_bytes,
                "json": {
                    "replay_id": item['replay_id'],
                    "timestamp": item['timestamp'],
                },
            })
    print()


if __name__ == "__main__":
    output_dir = Path(config.output_dir)

    create_dataset(
        "train",
        output_dir / "train-%06d.tar",
    )

    create_dataset(
        "eval",
        output_dir / "test-%06d.tar",
        shuffle=False
    )
