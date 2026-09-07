from pydantic import BaseModel


class PositioningModelConfig(BaseModel):

    # Number of lstm layers
    n_layers: int = 1

    embedding_size: int = 256

    # The max sequence length used in training/eval
    sequence_length: int = 100
