from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from houston_server_core.replayer import list_replayers, start_process, stop_replayer

replayer_router = APIRouter(prefix="/replayer", tags=["Replayer"])


class StartReplayerRequest(BaseModel):
    name: str = Field(
        ...,
        description="Logical name for this replayer run",
        example="test_run_1"
    )

    replay_ids: list[str] = Field(
        ...,
        description="List of replay IDs to process",
        example=["replay_01", "replay_02"]
    )


class ReplayerResponse(BaseModel):
    replayer_id: str
    name: str
    port: int
    replay_ids: list[str]
    start_time: float
    fabric_pid: int
    replay_pid: int


@replayer_router.get(
    "",
    response_model=list[ReplayerResponse],
    summary="List all active replayers"
)
def get_replayers():
    """Retrieve all running replayer metadata from the pid storage."""
    return list_replayers()


@replayer_router.post(
    "",
    response_model=ReplayerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new replayer process pair"
)
def create_replayer(payload: StartReplayerRequest):
    """Spawns background process pairs (Fabric server & Replayer) and records their execution metadata."""
    if not payload.replay_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one replay ID must be provided."
        )

    try:
        data = start_process(name=payload.name, replay_ids=payload.replay_ids)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start replayer processes: {str(e)}"
        )


@replayer_router.delete(
    "/{replayer_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Stop a replayer process pair"
)
def terminate_replayer(replayer_id: str):
    """Terminates fabric and replayer processes associated with the given replayer ID via SIGTERM."""
    replayers = list_replayers()
    target = next((r for r in replayers if r.get("replayer_id") == replayer_id), None)

    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replayer with ID '{replayer_id}' not found."
        )

    try:
        stop_replayer(replayer_id)
    except ProcessLookupError:
        # Gracefully handle cases where the process terminated prior to API call
        pass
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to kill process group for {replayer_id}: {str(e)}"
        )

    return None