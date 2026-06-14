import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cat_detector.camera import (
    reconcile_valve_on_startup,
    run_camera_loop,
    valve_watchdog_loop,
    _release_capture,
)
from cat_detector.config import settings
from cat_detector.database import init_db
from cat_detector.recording import start_buffer, stop_buffer
from cat_detector.routers.detections import router as detections_router
from cat_detector.routers.recordings import router as recordings_router
from cat_detector.routers.settings import router as settings_router
from cat_detector.routers.stream import router as stream_router
from cat_detector.routers.training import router as training_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Force valve closed on boot so a previous crash mid-open doesn't leave
    # water running. Best-effort: don't block startup if the call fails.
    try:
        await reconcile_valve_on_startup()
    except Exception:
        logger.exception("Valve reconcile on startup failed — continuing")
    task = asyncio.create_task(run_camera_loop())
    watchdog = asyncio.create_task(valve_watchdog_loop())
    start_buffer()
    logger.info("Cat detector service started")
    yield
    stop_buffer()
    for t in (task, watchdog):
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    _release_capture()
    logger.info("Cat detector service stopped")


app = FastAPI(title="Cat Detector", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(detections_router)
app.include_router(recordings_router)
app.include_router(settings_router)
app.include_router(stream_router)
app.include_router(training_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


def run():
    uvicorn.run("cat_detector.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
