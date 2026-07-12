import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi.responses import JSONResponse

from cat_detector.camera import (
    get_camera_status,
    get_last_process_age,
    reconcile_valve_on_startup,
    run_camera_loop,
    valve_watchdog_loop,
    _release_capture,
)
from cat_detector.config import settings
from cat_detector.database import init_db
from cat_detector.recording import get_buffer_status, start_buffer, stop_buffer
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


# Healthcheck thresholds. Anything OLDER than these in seconds is failing.
_HEALTH_DETECT_MAX_AGE_S = 30   # detect loop = 2s interval; 30s = 15 missed cycles
_HEALTH_FRAME_MAX_AGE_S = 30    # buffer should always have a fresh frame
_HEALTH_VALVE_MAX_OPEN_S = 120  # valve stuck open well past dwell + watchdog


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Deep health probe. Used by the external healthchecks.io heartbeat
    (via cron + healthcheck.sh) and Docker HEALTHCHECK. Returns 200 when
    every critical subsystem looks alive within its expected interval;
    503 otherwise — caller treats anything non-200 as failing."""
    from cat_detector import homeassistant
    detect_age = get_last_process_age()
    buf = get_buffer_status()
    cam_status = get_camera_status()
    failures: list[str] = []
    if detect_age is None or detect_age > _HEALTH_DETECT_MAX_AGE_S:
        failures.append(f"detect_loop_age={detect_age}")
    if buf.get("newest_age_s") is None or buf["newest_age_s"] > _HEALTH_FRAME_MAX_AGE_S:
        failures.append(f"buffer_newest_age={buf.get('newest_age_s')}")
    if cam_status not in ("connected", "unknown"):
        failures.append(f"camera_status={cam_status}")
    valve_state = None
    if homeassistant.is_configured():
        try:
            valve_state = await homeassistant.get_valve_state()
        except Exception:
            valve_state = "error"
    body = {
        "status": "ok" if not failures else "failing",
        "detect_loop_age_s": detect_age,
        "buffer": buf,
        "camera_status": cam_status,
        "valve_state": valve_state,
        "failures": failures,
    }
    return JSONResponse(body, status_code=200 if not failures else 503)


def run():
    uvicorn.run("cat_detector.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
