from typing import Literal

from pydantic import BaseModel, Field


class PositioningModelConfig(BaseModel):

    # Number of lstm layers
    n_layers: int = Field(1, description="Number of lstm layers")

    embedding_size: int = 64

    # The max sequence length used in training/eval
    sequence_length: int = Field(100, description="The max sequence length used in training/eval")

    include_camera: bool = False

    camera_height: int = 84

    camera_chunk_size: int = 16

    camera_type: Literal["rgb", "depth"] = "rgb"