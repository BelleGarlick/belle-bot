from pydantic import BaseModel, Field


class PositioningModelConfig(BaseModel):

    # Number of lstm layers
    n_layers: int = Field(1, description="Number of lstm layers")

    embedding_size: int = 256

    # The max sequence length used in training/eval
    sequence_length: int = Field(100, description="The max sequence length used in training/eval")
