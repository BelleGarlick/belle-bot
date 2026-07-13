from fastapi import UploadFile, APIRouter, HTTPException, Query, Form, File, Response
from houston.server.houston_server_core import models as core
from houston_server_persistence.models import Model

models_router = APIRouter(prefix="/models", tags=["Models"])


@models_router.post("/")
async def upload_model(
        file: UploadFile = File(...),
        name: str = Form(...),
        version: str = Form(...),
        tags: list[str] = Form(default_factory=list),
        description: str = Form(...),
) -> Model:
    if len(tags) == 1 and "," in tags[0]:
        tags = [t.strip() for t in tags[0].split(",")]

    return core.upload_model(
        name=name,
        version=version,
        tags=tags,
        description=description,
        upload=file,
    )


@models_router.get("/")
async def list_models(
    page: int | None = Query(None),
):
    models, count = core.query_models(page or 0)
    return {
        "models": models,
        "total": count
    }


@models_router.get("/{model_id}")
async def get_model_file(model_id: str):
    model = core.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    content = core.get_model_object(model)
    if not content:
        raise HTTPException(status_code=404, detail="Model not found")

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={model.name}"}
    )


@models_router.get("/{model_id}/info", response_model=Model)
async def get_model_info(model_id: str):
    model = core.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
