from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from houston_server_core import datasets as core
from houston_server_persistence.dataset import Dataset
from pydantic import BaseModel

dataset_router = APIRouter(prefix="/datasets", tags=["Datasets"])


class DatasetListResponse(BaseModel):
    datasets: list[Dataset]
    total: int


@dataset_router.post(
    "/",
    response_model=Dataset,
    summary="Upload a new dataset",
    description="Uploads a WebDataset format file and creates a new dataset record with tags and description.",
)
async def upload_dataset(
    file: UploadFile = File(..., description="The dataset file to upload (tar/tar.gz)"),
    tags: list[str] = Form(
        default_factory=list,
        description="List of tags or a comma-separated string of tags",
    ),
    description: str | None = Form(
        default=None, description="Detailed description of the dataset"
    ),
    path: str | None = Form(
        default=None,
        description="Custom path/name for the dataset (e.g. 'vision/encoder/v1/train')",
    ),
) -> Dataset:
    """
    Upload a new dataset.

    If tags contains a single string with commas, it will be split into multiple tags.
    """
    if len(tags) == 1 and "," in tags[0]:
        tags = [t.strip() for t in tags[0].split(",")]

    return core.upload_dataset(
        tags=tags, description=description, upload=file, path_override=path
    )


@dataset_router.get(
    "/",
    response_model=DatasetListResponse,
    summary="List datasets",
    description="Retrieves a paginated list of datasets, optionally filtered by tags.",
)
async def list_datasets(
    page: int | None = Query(None, description="Page number for pagination"),
    tags: list[str] | None = Query(
        None, description="Filter datasets by one or more tags"
    ),
) -> DatasetListResponse:
    """List all available datasets with optional tag filtering."""
    datasets, count = core.query_datasets(page or 0, tags=tags)
    return DatasetListResponse(datasets=datasets, total=count)


@dataset_router.get(
    "/{dataset_id}",
    summary="Download dataset file",
    description="Downloads the actual dataset file associated with the given ID.",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"description": "Dataset not found"},
    },
)
async def get_dataset_file(dataset_id: str) -> FileResponse:
    """Download the dataset file."""
    dataset = core.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = core.get_dataset_file_path(dataset)
    if not file_path:
        raise HTTPException(status_code=404, detail="Dataset file not found")

    filename = dataset.path or f"{dataset.dataset_id}.tar"
    if not filename.endswith((".tar", ".tar.gz", ".tgz")):
        # Ensure it has a reasonable extension if we can't tell
        if "." not in filename:
            filename += ".tar"

    return FileResponse(
        path=file_path, media_type="application/octet-stream", filename=filename
    )


@dataset_router.get(
    "/{dataset_id}/info",
    response_model=Dataset,
    summary="Get dataset information",
    description="Retrieves the metadata for a specific dataset.",
)
async def get_dataset_info(dataset_id: str) -> Dataset:
    """Get metadata for a specific dataset."""
    dataset = core.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@dataset_router.put(
    "/{dataset_id}",
    response_model=Dataset,
    summary="Update dataset",
    description="Updates the metadata (tags, description, etc.) for an existing dataset.",
)
async def update_dataset(dataset_id: str, body: Dataset) -> Dataset:
    """Update dataset metadata."""
    dataset = core.update_dataset(dataset_id, body)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@dataset_router.delete(
    "/{dataset_id}",
    status_code=204,
    summary="Delete dataset",
    description="Deletes a dataset and its associated file from storage.",
)
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    core.delete_dataset(dataset_id)
    return Response(status_code=204)
