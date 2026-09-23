import base64
import io
import json
from pathlib import Path
from typing import Literal

from PIL import Image
from matplotlib import pyplot as plt

import cv2


PATH = "/Users/belle/Developer/belle-bot/belle_bot/mapping/positioning/training/environment/replays_cache/049e17b1-9646-4fc6-a41c-e7bf651b29ae.txt"


def create_dataset():
    with open(PATH) as file:
        lines = file.read().split("\n")

    for line in lines[7000:]:
        split_tokens = line.split(",")
        if len(split_tokens) < 3:
            continue

        stream = split_tokens[0]
        timestamp = float(split_tokens[1])
        data = json.loads(",".join(split_tokens[2:]))

        import numpy as np
        if stream == "sensors/camera":
            main_image = Image.open(io.BytesIO(base64.b64decode(data['rgb'])))
            
            depth_raw_encoded = base64.b64decode(data['depth_raw'])
            
            # Try to decode as PNG first (new format)
            depth_raw_buffer = cv2.imdecode(np.frombuffer(depth_raw_encoded, np.uint8), cv2.IMREAD_UNCHANGED)
            
            if depth_raw_buffer is None:
                # Fallback to raw buffer (old format)
                # 320 * 240 * 2 bytes = 153600
                try:
                    depth_raw_buffer_b = np.frombuffer(depth_raw_encoded, dtype=np.uint16).reshape((320, 240))
                    depth_raw_buffer = cv2.resize(depth_raw_buffer_b, (320, 240))

                    _, encoded_png = cv2.imencode('.png', depth_raw_buffer)
                    restored_depth_data = cv2.imdecode(encoded_png, cv2.IMREAD_UNCHANGED)
                    # breakpoint()

                    depth_visual = cv2.normalize(depth_raw_buffer, None, 0, 255, cv2.NORM_MINMAX)
                    depth_visual = np.uint8(depth_visual)

                    cv2.imwrite("depth.png", depth_visual)
                    # with open("test.png", "wb") as file:
                    #     file.write(depth_visual)

                        # store depth and depth preview, the depth is reconsitrctured like this and the preview si just a jpeg like the other image

                    # depth_raw_buffer = np.resize(depth_raw_buffer, (240, 320))
                except Exception as e:
                    print(f"Failed to decode depth data at timestamp {timestamp}: {e}")
                    continue
            
            plt.subplot(131), plt.imshow(main_image)
            plt.subplot(132), plt.imshow(depth_raw_buffer)
            plt.subplot(133), plt.imshow(depth_visual)
            plt.show()

            _, color_buffer = cv2.imencode('.png', depth_raw_buffer)

            # todo further testing

            # # 480/320
            # all_items.append({
            #     "replay_id": replay_id,
            #     "timestamp": timestamp,
            #     "rgb": data['rgb'],
            #     "depth": data['depth']
            # })


if __name__ == "__main__":
    # output_dir = Path(config.output_dir)

    create_dataset(
        # "train",
        # output_dir / "train-%06d.tar",
    )

    # create_dataset(
        # "eval",
        # output_dir / "test-%06d.tar",
        # shuffle=False
    # )
