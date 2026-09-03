import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from houston_server_api import routes
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

frontend_dist = Path(__file__).parent / "../../frontend/dist"

app = FastAPI(
    title="Houston API",
    middleware=middleware,
    generate_unique_id_function=lambda route: route.name
)


app.include_router(routes.replay_router)
app.include_router(routes.replayer_router)
app.include_router(routes.models_router)
# app.include_router(dataset_router)

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dist, "index.html"))

app.mount("/", StaticFiles(directory=frontend_dist), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
