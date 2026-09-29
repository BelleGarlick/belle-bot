from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from houston_server_core import models as core
from houston_server_persistence.models import Model
from pydantic import BaseModel

models_router = APIRouter(prefix="/models", tags=["Models"])


class ModelListResponse(BaseModel):
    models: list[Model]
    total: int


@models_router.post(
    "/",
    response_model=Model,
    summary="Upload a new model",
    description="Uploads a model file and creates a new model record with name, version, and metadata.",
)
async def upload_model(
    file: UploadFile = File(..., description="The model file to upload"),
    name: str = Form(..., description="Name of the model"),
    version: str = Form(..., description="Version string for the model"),
    tags: list[str] = Form(
        default_factory=list,
        description="List of tags or a comma-separated string of tags",
    ),
    description: str = Form(..., description="Detailed description of the model"),
) -> Model:
    """
    Upload a new model.

    If tags contains a single string with commas, it will be split into multiple tags.
    """
    if len(tags) == 1 and "," in tags[0]:
        tags = [t.strip() for t in tags[0].split(",")]

    return core.upload_model(
        name=name,
        version=version,
        tags=tags,
        description=description,
        upload=file,
    )


@models_router.get(
    "/",
    response_model=ModelListResponse,
    summary="List models",
    description="Retrieves a paginated list of models.",
)
async def list_models(
    page: int | None = Query(None, description="Page number for pagination"),
) -> ModelListResponse:
    """List all available models."""
    models, count = core.query_models(page or 0)
    return ModelListResponse(models=models, total=count)


@models_router.get(
    "/{model_id}",
    summary="Download model file",
    description="Downloads the actual model file associated with the given ID.",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"description": "Model not found"},
    },
)
async def get_model_file(model_id: str) -> Response:
    """Download the model file."""
    model = core.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    content = core.get_model_object(model)
    if not content:
        raise HTTPException(status_code=404, detail="Model not found")

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={model.name}"},
    )


@models_router.get(
    "/{model_id}/info",
    response_model=Model,
    summary="Get model information",
    description="Retrieves the metadata for a specific model.",
)
async def get_model_info(model_id: str) -> Model:
    """Get metadata for a specific model."""
    model = core.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
