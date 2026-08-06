from fastapi import UploadFile, APIRouter, HTTPException, Query, Form, File, Response
from houston_server_core import replays as core
from houston_server_persistence.replay import Replay
from pydantic import BaseModel

replay_router = APIRouter(prefix="/replays", tags=["Replays"])


class ReplayListResponse(BaseModel):
    replays: list[Replay]
    total: int


@replay_router.post("/", response_model=Replay)
async def upload_replay(
        file: UploadFile = File(...),
        filename: str | None = Form(default=None),
        platform: str | None = Form(default=None),
        tags: list[str] = Form(default_factory=list),
        description: str | None = Form(default=None),
        permanent: bool = Form(default=False),
) -> Replay:
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


@replay_router.get("/", response_model=ReplayListResponse)
async def list_replays(
    page: int | None = Query(None),
) -> ReplayListResponse:
    replays, count = core.query_replays(page or 0)
    return ReplayListResponse(
        replays=replays,
        total=count
    )


@replay_router.get("/{replay_id}")
async def get_replay_file(replay_id: str) -> Response:
    replay = core.get_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")

    content = core.get_replay_object(replay)
    if not content:
        raise HTTPException(status_code=404, detail="Replay not found")

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={replay.filename}"}
    )


@replay_router.get("/{replay_id}/info", response_model=Replay)
async def get_replay_info(replay_id: str) -> Replay:
    replay = core.get_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay


@replay_router.put("/{replay_id}", response_model=Replay)
async def update_replay(replay_id: str, body: Replay) -> Replay:
    replay = core.update_replay(replay_id, body)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay
