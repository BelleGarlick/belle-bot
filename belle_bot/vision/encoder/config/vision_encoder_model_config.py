from pydantic import BaseModel


class VisionEncoderModelConfig(BaseModel):

    embedding_size: int = 27*27
