from fastapi import FastAPI
from houston_server_api import routes

app = FastAPI()

app.include_router(routes.replay_router)
app.include_router(routes.models_router)
# app.include_router(dataset_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
