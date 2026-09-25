from pydantic import BaseModel, Field

from houston.client.py.config import HoustonConfig


class VisionEncoderDatasetCreationConfig(BaseModel):

    houston: HoustonConfig = HoustonConfig()

    max_partition_size: int = Field(100_000, description="The max number of items per dataset item.")

    output_dir: str = Field("vision-encoder/v1", description="The output file path where the dataset will be created")
