from pydantic import BaseModel, Field


class HoustonConfig(BaseModel):

    endpoint: str = Field("http://localhost:8080", description="Houston endpoint URL")
