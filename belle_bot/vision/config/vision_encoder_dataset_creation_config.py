from typing import Literal

from pydantic import BaseModel, Field

from houston.client.py.config import HoustonConfig


class VisionEncoderDatasetCreationMaskConfig(BaseModel):

    count: int = Field(0, description="The number of masks to create which images sample from. 0 disables mask generation.")

    size: int = Field(448, description="The size of each mask.")


class VisionEncoderDatasetCreationConfig(BaseModel):

    houston: HoustonConfig = HoustonConfig()

    mask: VisionEncoderDatasetCreationMaskConfig = VisionEncoderDatasetCreationMaskConfig()

    max_partition_size: int = Field(100_000, description="The max number of items per dataset item.")

    path: str = Field("train-%06d.tar", description="The output file path where the dataset will be created")

    frequency_map: str | None = Field(None, description="If given, items will be downsampled based on the frequency of their cosine similar neighbours. Doing so allows us to downsample similar frames.")

    subset: Literal["train", "test"] = Field("train", description="Which split of data to create..")
