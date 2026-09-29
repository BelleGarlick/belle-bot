import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from houston_server_api import routes
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

frontend_dist = Path(__file__).parent / "../../frontend/dist"

app = FastAPI(
    title="Houston API",
    middleware=middleware,
    generate_unique_id_function=lambda route: route.name,
)

api_router = APIRouter(prefix="/api")
api_router.include_router(routes.replay_router)
api_router.include_router(routes.replayer_router)
api_router.include_router(routes.models_router)
api_router.include_router(routes.dataset_router)
app.include_router(api_router)


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404 and not request.url.path.startswith("/api/"):
        return FileResponse(os.path.join(frontend_dist, "index.html"))
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dist, "index.html"))


app.mount("/", StaticFiles(directory=frontend_dist), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
