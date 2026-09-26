from pydantic import BaseModel, Field


class HoustonConfig(BaseModel):

    enabled: bool = Field(True, description="Enables Houston training")

    endpoint: str = Field("http://houston:8080", description="Houston endpoint URL")
