import cv2
import numpy as np



def letterbox_image(frame: np.ndarray, size: int):
    # resize the image
    frame_shape = frame.shape
    rescale_ratio = frame_shape[1] / size
    new_height = int(frame_shape[0] / rescale_ratio)
    frame = cv2.resize(frame, (size, new_height))

    # apply letter box
    vertical_padding_offset = (size - frame.shape[0]) // 2
    vertical_padding = np.zeros((vertical_padding_offset, size, 3), dtype=np.int8)

    return (
        np.concatenate([vertical_padding, frame, vertical_padding], axis=0).astype(np.uint8),
        rescale_ratio,
        vertical_padding_offset
    )

