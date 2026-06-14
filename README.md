# Cat Detector

A surveillance camera service that watches an outdoor garden sofa and fires a webhook when cats are detected — but not when humans are also present.

## Features

- Real-time RTSP/USB camera stream monitoring with exponential backoff reconnection
- YOLOv8-based cat, dog, and human detection (dog as misclassification safety net)
- Webhook alert with cooldown (no spam) — triggers on cats/dogs, suppressed when humans present
- Live MJPEG video stream with detection overlays (`/api/stream`)
- Web UI with tabbed Live View and Detection History
- Per-detection bounding box persistence (class, confidence, coordinates)
- True positive / false positive / false negative labeling from the UI
- "Report Missed Cat" button for live snapshot false negative capture
- Fine-tuning pipeline integration (labeled frames feed into training)
- Environment-based configuration with startup validation

## Requirements

- Python 3.11+
- A camera accessible via RTSP URL, HTTP stream, or local device index

## Setup

```bash
# 1. Clone / enter the project
cd cat-detector

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env — set CAMERA_URL and WEBHOOK_URL at minimum

# 5. Run
cat-detector
# or: python -m cat_detector.main
```

Open http://localhost:8000 to see the web UI.

## Configuration

All settings are read from `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `CAMERA_URL` | `0` | RTSP URL, HTTP stream URL, or device index |
| `DETECTION_MODEL` | `yolov8m.pt` | YOLOv8 model size (n/s/m/l/x) |
| `DETECTION_CONFIDENCE` | `0.35` | Minimum detection confidence |
| `CAMERA_RECONNECT_MAX_DELAY` | `60` | Max backoff delay (seconds) on camera disconnect |
| `DETECTION_INTERVAL_SECONDS` | `5` | Seconds between frame samples |
| `WEBHOOK_COOLDOWN_SECONDS` | `300` | Minimum gap between webhook calls |
| `WEBHOOK_URL` | — | Target URL for cat alerts |
| `WEBHOOK_SECRET` | — | Optional secret sent as `X-Webhook-Secret` |
| `FRAMES_DIR` | `./data/frames` | Where annotated frames are stored |

## Webhook Payload

```json
{
  "event": "cat_detected",
  "detection_id": 42,
  "cat_count": 2,
  "max_confidence": 0.9123
}
```

The webhook is **not called** when:
- No cats are detected
- A human is also present in the frame
- The cooldown period since the last call has not elapsed

## Detection Logic

The service uses YOLOv8 (COCO classes: `cat=15`, `dog=16`, `person=0`). On each sampled frame:

1. Run inference
2. Count cats and humans above the confidence threshold
3. If `cats > 0` and `humans == 0` and cooldown elapsed → fire webhook
4. Always store the annotated frame and detection metadata

## Web UI

Browse to `http://localhost:8000` to:
- **Live View** — real-time camera feed with detection bounding boxes overlaid
- **Detection History** — browse past detections with annotated frames, timestamps, and confidence scores
- Label each detection as **Correct** (true positive), **Wrong** (false positive), or **Missed Cat** (false negative)
- **Report Missed Cat** — capture a snapshot from the live feed when you see a cat the model missed
- Trigger fine-tuning on labeled data (true positives + false negatives = positive examples)

## Running Tests

```bash
pytest
```

Unit tests in `tests/test_detector.py` cover the trigger logic without requiring a camera or GPU.

## Fine-tuning

The fine-tuning trigger is a stub in `src/cat_detector/routers/training.py` (`_run_finetuning`). To implement:

1. Export labeled frames to a YOLO dataset layout
2. Call `YOLO(settings.detection_model).train(data="dataset.yaml", epochs=10)`
3. Update `settings.detection_model` to point at the new weights

## Project Structure

```
cat-detector/
├── src/cat_detector/
│   ├── main.py             # FastAPI app + service entry point
│   ├── camera.py           # Camera loop + shared capture (thread-safe)
│   ├── detector.py         # YOLOv8 inference + trigger logic
│   ├── webhook.py          # Webhook HTTP call
│   ├── models.py           # SQLAlchemy ORM models
│   ├── database.py         # DB session setup
│   ├── config.py           # Pydantic settings with validation
│   └── routers/
│       ├── detections.py   # Detection CRUD, labeling, report-missed
│       ├── stream.py       # MJPEG live streaming endpoint
│       └── training.py     # Training status/trigger endpoints
├── templates/index.html    # Web UI (tabs: Live View + Detection History)
├── tests/                  # Unit + integration tests
├── docs/adr/               # Architecture Decision Records
├── docs/prd.md             # Product Requirements Document
├── infra/                  # Terraform (Azure App Service, SQL, Key Vault)
├── deliverables/           # Certification submission artifacts
├── .claude/                # Custom skills, agents, hooks
├── .env.example
├── LICENSE                 # AGPL-3.0
└── pyproject.toml
```
