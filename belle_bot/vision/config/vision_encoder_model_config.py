from pydantic import BaseModel, Field


class VisualCortexEncoderModelConfig(BaseModel):

    file_path: str | None = Field(None, description="The file path for the visual cortex encoder model")
    frozen: bool = Field(default=False, description="Whether this component has been frozen for training")


class VisualCortexDecoderModelConfig(BaseModel):

    file_path: str | None = Field(None, description="The file path for the visual cortex decoder model")


class VisualCortexSegmentationModelConfig(BaseModel):

    file_path: str | None = Field(None, description="The file path for the visual cortex segmentation model")


class VisualCortexLanguageModelConfig(BaseModel):

    file_path: str | None = Field(None, description="The file path for the visual cortex language model")


class VisualCortexDetectionModelConfig(BaseModel):

    file_path: str | None = Field(None, description="The file path for the visual cortex detection model")


class VisionEncoderModelConfig(BaseModel):

    embedding_size: int = 32*32

    encoder: VisualCortexEncoderModelConfig = VisualCortexEncoderModelConfig()
    decoder: VisualCortexDecoderModelConfig = VisualCortexDecoderModelConfig()
    segmentation: VisualCortexSegmentationModelConfig = VisualCortexSegmentationModelConfig()
    language: VisualCortexLanguageModelConfig = VisualCortexLanguageModelConfig()
    detection: VisualCortexDetectionModelConfig = VisualCortexDetectionModelConfig()
