from pydantic import BaseModel, Field


class HoustonConfig(BaseModel):

    endpoint: str = Field("http://192.168.0.182:8080", description="Houston endpoint URL")
