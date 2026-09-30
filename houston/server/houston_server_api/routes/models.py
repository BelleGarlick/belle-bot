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
    description="Uploads a model file (or .zip for a directory) and creates a new model record with name, version, and metadata.",
)
async def upload_model(
    file: UploadFile = File(..., description="The model file to upload (.zip for directories)"),
    name: str = Form(..., description="Name of the model (its path)"),
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
    name: str | None = Query(None, description="Filter by model name"),
    tags: list[str] | None = Query(None, description="Filter by tags"),
) -> ModelListResponse:
    """List all available models, optionally filtered by name or tags."""
    if tags and len(tags) == 1 and "," in tags[0]:
        tags = [t.strip() for t in tags[0].split(",")]

    models, count = core.query_models(page or 0, tags=tags, name=name)
    return ModelListResponse(models=models, total=count)


@models_router.get(
    "/{model_id}/files",
    response_model=list[str],
    summary="List files in a model directory",
    description="Returns a list of all files contained within the model's directory.",
)
async def list_model_files(model_id: str) -> list[str]:
    """List files in the model directory."""
    model = core.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return core.list_model_files(model)


@models_router.get(
    "/{model_id}/file/{file_path:path}",
    summary="Download a specific file from a model",
    description="Downloads a specific file within a model directory using its relative path.",
)
async def get_model_file(model_id: str, file_path: str) -> Response:
    """Download a specific file from the model."""
    model = core.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    content = core.get_model_file(model, file_path)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")

    filename = file_path.split("/")[-1]
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@models_router.get(
    "/{model_id}",
    summary="Download model",
    description="Downloads the model file. For directory models, this returns a zip of the directory.",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"description": "Model not found"},
    },
)
async def download_model(model_id: str) -> Response:
    """Download the model file or zip."""
    model = core.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    content = core.get_model_object(model)
    if content is None:
        raise HTTPException(status_code=404, detail="Model file not found")

    filename = model.name
    if model.is_dir and not filename.endswith(".zip"):
        filename += ".zip"

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
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


@models_router.post(
    "/{model_id}/file",
    response_model=Model,
    summary="Upload an individual file to a model",
    description="Uploads a single file to a model version. If the model was previously a single-file model, it will be converted to a directory model.",
)
async def upload_model_file(
    model_id: str,
    file: UploadFile = File(..., description="The file to upload"),
    relative_path: str = Form(..., description="The relative path where the file should be stored"),
) -> Model:
    """Upload an individual file to an existing model."""
    try:
        return core.add_model_file(
            model_id=model_id,
            relative_path=relative_path,
            upload=file,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
