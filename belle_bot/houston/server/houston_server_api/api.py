from fastapi import FastAPI
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


app = FastAPI(
    title="Houston API",
    middleware=middleware,
    generate_unique_id_function=lambda route: route.name
)


app.include_router(routes.replay_router)
app.include_router(routes.replayer_router)
app.include_router(routes.models_router)
# app.include_router(dataset_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)
