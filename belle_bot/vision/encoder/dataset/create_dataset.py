import base64
import json
from typing import Literal

import random
import webdataset as wds

from houston.client.py import replays
from belle_bot.vision.encoder.config.vision_encoder_config import VisionEncoderConfig

config = VisionEncoderConfig()


def get_replay_ids(subset: Literal["training", "testing"] | None):
    filter = []
    if subset:
        filter += [subset]

    replay_ids = replays.query_replays(
        config.houston,
        page=0,
        tags=filter
    )['replays']

    return sorted([x["replay_id"] for x in replay_ids])


def create_dataset(subset: Literal["training", "testing"], rgb_pattern: str, depth_pattern: str, shuffle=True):
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

            if stream == "sensors/camera":
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
        wds.ShardWriter(rgb_pattern, maxcount=100_000) as rgb_writer,
        wds.ShardWriter(depth_pattern, maxcount=100_000) as depth_writer
    ):
        for item_count, item in enumerate(all_items):
            print(f"\r{subset} Writing {item_count}/{len(all_items)}", end="")

            try:
                rgb_bytes = base64.b64decode(item['rgb'])
            except Exception:
                rgb_bytes = item['rgb']
                
            rgb_writer.write({
                "__key__": f"{item_count}",
                "jpg": rgb_bytes,
                "json": {
                    "replay_id": item['replay_id'],
                    "timestamp": item['timestamp'],
                },
            })

            try:
                depth_bytes = base64.b64decode(item['depth'])
            except Exception:
                depth_bytes = item['depth']

            depth_writer.write({
                "__key__": f"{item_count}",
                "jpg": depth_bytes,
                "json": {
                    "replay_id": item['replay_id'],
                    "timestamp": item['timestamp'],
                },
            })
    print()


if __name__ == "__main__":
    create_dataset(
        "training",
        "vision-encoder/v1/rgb/train-%06d.tar",
        "vision-encoder/v1/depth/train-%06d.tar"
    )

    create_dataset(
        "testing",
        "vision-encoder/v1/rgb/test-%06d.tar",
        "vision-encoder/v1/depth/test-%06d.tar",
        shuffle=False
    )
