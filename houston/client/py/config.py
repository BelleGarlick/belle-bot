from pydantic import BaseModel, Field


class HoustonConfig(BaseModel):

    endpoint: str = Field("http://houston:8080", description="Houston endpoint URL")
