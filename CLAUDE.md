# Cat Detector — Project Notes for Claude

## What this project does

Watches an RTSP/USB camera stream pointed at an outdoor garden sofa. When one or more cats are detected (and no humans are present), it fires a configurable webhook. It also exposes a web UI for reviewing past detections, labeling them, and triggering fine-tuning.

## Tech stack

- **FastAPI** — web framework + API
- **YOLOv8 (ultralytics)** — object detection (COCO classes: `cat=15`, `person=0`)
- **OpenCV** — camera frame capture
- **SQLAlchemy + SQLite** — detection persistence
- **Jinja2** — web UI templates (vanilla JS, no frontend build step)
- **httpx** — async webhook calls
- **Pydantic Settings** — `.env`-based config

## Key design decisions

- Camera is sampled on a fixed interval (not continuous streaming) to keep CPU/memory manageable. Change `DETECTION_INTERVAL_SECONDS` to tune.
- Webhook cooldown prevents alert spam. Change `WEBHOOK_COOLDOWN_SECONDS`.
- YOLOv8 model size is configurable (`yolov8n` → fastest, `yolov8x` → most accurate).
- Fine-tuning is intentionally a stub — implement `_run_finetuning` in `routers/training.py`.

## Entry points

- `cat_detector.main:app` — FastAPI ASGI app
- `cat_detector.main:run` — uvicorn runner (CLI entry point `cat-detector`)
- `cat_detector.camera:run_camera_loop` — background asyncio task, started in lifespan

## Where things live

| Concern | File |
|---|---|
| Detection inference + trigger logic | `src/cat_detector/detector.py` |
| Camera loop / orchestration | `src/cat_detector/camera.py` |
| Webhook HTTP call | `src/cat_detector/webhook.py` |
| DB schema | `src/cat_detector/models.py` |
| REST API — detections | `src/cat_detector/routers/detections.py` |
| REST API — training | `src/cat_detector/routers/training.py` |
| Web UI | `templates/index.html` |
| Settings | `src/cat_detector/config.py` + `.env` |

## Test data

Place test frames in `tests/frames/`. See `tests/frames/README.md` for naming convention. The unit tests in `tests/test_detector.py` are camera-free and test only the trigger logic.

## Common tasks

**Add a new detection field** → update `models.py`, add a migration or `drop + recreate` (SQLite), update `DetectionOut` in `routers/detections.py`.

**Switch to a vision LLM backend** → replace `detector.py`'s `detect()` function. The rest of the pipeline (camera loop, webhook, storage) is model-agnostic.

**Deploy as a systemd service** → `cat-detector` is the CLI entry point after `pip install -e .`. Point `WorkingDirectory` at the project root so `.env` and `templates/` are found.
