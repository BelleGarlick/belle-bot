from fastapi import UploadFile, APIRouter, HTTPException, Query, Form, File, Response
from houston_server_core import replays as core
from houston_server_persistence.replay import Replay

replay_router = APIRouter(prefix="/replays", tags=["Replays"])


@replay_router.post("/")
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


@replay_router.get("/")
async def list_replays(
    page: int | None = Query(None),
):
    replays, count = core.query_replays(page or 0)
    return {
        "replays": replays,
        "total": count
    }


@replay_router.get("/{replay_id}")
async def get_replay_file(replay_id: str):
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


@replay_router.get("/{replay_id}/info")
async def get_replay_info(replay_id: str):
    replay = core.get_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay
