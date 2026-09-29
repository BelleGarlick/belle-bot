from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from houston_server_core import replays as core
from houston_server_persistence.replay import Replay
from pydantic import BaseModel

replay_router = APIRouter(prefix="/replays", tags=["Replays"])


class ReplayListResponse(BaseModel):
    replays: list[Replay]
    total: int


@replay_router.post(
    "",
    response_model=Replay,
    summary="Upload a new replay",
    description="Uploads a replay file and creates a new replay record with metadata.",
)
async def upload_replay(
    file: UploadFile = File(..., description="The replay file to upload"),
    filename: str | None = Form(
        default=None, description="Original filename of the replay"
    ),
    platform: str | None = Form(
        default=None, description="Platform where the replay was recorded"
    ),
    tags: list[str] = Form(
        default_factory=list,
        description="List of tags or a comma-separated string of tags",
    ),
    description: str | None = Form(
        default=None, description="Detailed description of the replay"
    ),
    permanent: bool = Form(
        default=False, description="Whether to mark the replay as permanent"
    ),
) -> Replay:
    """
    Upload a new replay.

    If tags contains a single string with commas, it will be split into multiple tags.
    """
    if len(tags) == 1 and "," in tags[0]:
        tags = [t.strip() for t in tags[0].split(",")]

    return core.upload_replay(
        filename=filename,
        platform=platform,
        permanent=permanent,
        tags=tags,
        description=description,
        upload=file,
    )


@replay_router.get(
    "",
    response_model=ReplayListResponse,
    summary="List replays",
    description="Retrieves a paginated list of replays, optionally filtered by tags.",
)
async def list_replays(
    page: int | None = Query(None, description="Page number for pagination"),
    tags: list[str] | None = Query(
        None, description="Filter replays by one or more tags"
    ),
) -> ReplayListResponse:
    """List all available replays with optional tag filtering."""
    replays, count = core.query_replays(page or 0, tags=tags)
    return ReplayListResponse(replays=replays, total=count)


@replay_router.get(
    "/{replay_id}",
    summary="Download replay file",
    description="Downloads the actual replay file associated with the given ID.",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"description": "Replay not found"},
    },
)
async def get_replay_file(replay_id: str) -> FileResponse:
    """Download the replay file."""
    replay = core.get_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")

    file_path = core.get_replay_object(replay)
    if not file_path:
        raise HTTPException(status_code=404, detail="Replay not found")

    filename = replay.filename or replay.path or f"{replay.replay_id}.txt"

    return FileResponse(
        path=file_path, media_type="application/octet-stream", filename=filename
    )


@replay_router.get(
    "/{replay_id}/info",
    response_model=Replay,
    summary="Get replay information",
    description="Retrieves the metadata for a specific replay.",
)
async def get_replay_info(replay_id: str) -> Replay:
    """Get metadata for a specific replay."""
    replay = core.get_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay


@replay_router.put(
    "/{replay_id}",
    response_model=Replay,
    summary="Update replay",
    description="Updates the metadata (tags, description, etc.) for an existing replay.",
)
async def update_replay(replay_id: str, body: Replay) -> Replay:
    """Update replay metadata."""
    replay = core.update_replay(replay_id, body)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay


@replay_router.delete(
    "/{replay_id}",
    status_code=204,
    summary="Delete replay",
    description="Deletes a replay and its associated file from storage.",
)
async def delete_replay(replay_id: str):
    """Delete a replay."""
    core.delete_replay(replay_id)
    return Response(status_code=204)
