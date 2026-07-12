from fastapi import FastAPI
from houston.server.houston_server_api.routes.replays import replay_router

app = FastAPI()

app.include_router(replay_router)
# app.include_router(models_router)
# app.include_router(dataset_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
