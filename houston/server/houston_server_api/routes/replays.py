import datetime

from fastapi import UploadFile, APIRouter, HTTPException, Query, Form, File, Response
from houston.server.houston_server_core import replays as core
from houston_server_persistence.replay import Replay

replay_router = APIRouter(prefix="/replays", tags=["Replays"])


@replay_router.post("/")
async def upload_replay(
        file: UploadFile = File(...),
        filename: str = Form(...),
        firmware_version: str = Form(...),
        physical_device: str = Form(...),
        tags: list[str] = Form(default_factory=list),
        description: str = Form(...),
        permanent: bool = Form(...),
        start_time: datetime.datetime = Form(...),
        end_time: datetime.datetime = Form(...)
) -> Replay:
    if len(tags) == 1 and "," in tags[0]:
        tags = [t.strip() for t in tags[0].split(",")]

    return core.upload_replay(
        filename=filename,
        firmware_version=firmware_version,
        physical_device=physical_device,
        permanent=permanent,
        start_time=start_time,
        end_time=end_time,
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
