from fastapi import UploadFile, APIRouter, HTTPException, Query, Form, File, Response
from pydantic import BaseModel
from houston.server.houston_server_core import replays as core

replay_router = APIRouter(prefix="/replays", tags=["Replays"])

class ReplayInfo(BaseModel):
    id: int
    filename: str
    description: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    permanent: bool
    tags: list[str]

@replay_router.post("/", response_model=ReplayInfo)
async def upload_replay(
    file: UploadFile = File(...),
    description: str | None = Form(None),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
    permanent: bool = Form(False),
    tags: list[str] = Form([])
):
    content = await file.read()
    
    # Handle tags if they are sent as a single comma-separated string or multiple Form entries
    # FastAPI usually handles multiple Form entries as a list, but let's be safe
    if len(tags) == 1 and "," in tags[0]:
        tags = [t.strip() for t in tags[0].split(",")]

    replay_id = core.upload_replay(
        filename=file.filename,
        content=content,
        description=description,
        start_time=start_time,
        end_time=end_time,
        permanent=permanent,
        tags=tags
    )
    
    replay = core.get_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=500, detail="Failed to retrieve uploaded replay")
        
    return replay

@replay_router.get("/", response_model=list[ReplayInfo])
async def list_replays(
    tag: str | None = Query(None),
    start_time: float | None = Query(None),
    end_time: float | None = Query(None),
    permanent: bool | None = Query(None)
):
    replays = core.list_replays(
        tag=tag,
        start_time=start_time,
        end_time=end_time,
        permanent=permanent
    )
    return replays

@replay_router.get("/{replay_id}")
async def get_replay_file(replay_id: int):
    replay = core.get_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
        
    return Response(
        content=replay["content"],
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={replay['filename']}"}
    )

@replay_router.get("/{replay_id}/info", response_model=ReplayInfo)
async def get_replay_info(replay_id: int):
    replay = core.get_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay

# todo add in replay listener
