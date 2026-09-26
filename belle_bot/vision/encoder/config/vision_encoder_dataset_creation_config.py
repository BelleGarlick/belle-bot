from pydantic import BaseModel, Field

from houston.client.py.config import HoustonConfig


class VisionEncoderDatasetCreationConfig(BaseModel):

    houston: HoustonConfig = HoustonConfig()

    max_partition_size: int = Field(100_000, description="The max number of items per dataset item.")

    output_dir: str = Field("/run/media/belle/Houston/datasets/vision-encoder/v1", description="The output file path where the dataset will be created")

    frequency_map: str | None = Field(None, description="If given, items will be downsampled based on the frequency of their cosine similar neighbours. Doing so allows us to downsample similar frames.")
