# ADR-0001: Technology Stack Choices

**Status:** Accepted  
**Date:** 2026-04-06  
**Author:** Cat Detector Team

---

## Context

We need to choose a runtime language, web framework, vision model, and persistence layer for a service that:

- Samples frames from an RTSP/USB camera on a fixed interval
- Runs object detection to identify cats and humans
- Fires an HTTP webhook on cat-only detections
- Serves a labeling UI and REST API
- Runs as a long-lived background service on modest hardware (e.g. a Raspberry Pi 5 or a small home server)

Key constraints:
- Single developer, training-project scope — low operational overhead is more important than scalability
- Must run without a GPU (or degrade gracefully when one is present)
- Detection must be fast enough to process one frame every 5 seconds on CPU

---

## Options Considered

### Language

| Option | Pros | Cons |
|---|---|---|
| **Python** | Best ML/CV ecosystem; ultralytics, OpenCV, FastAPI are all first-class | GIL limits true threading; slower than Go/Rust for pure IO |
| Go | Fast, single binary | Almost no native ML tooling; would need to shell out to Python anyway |
| Node.js | Good async IO | No serious CV ecosystem; TensorFlow.js is immature for this use case |

### Web Framework

| Option | Pros | Cons |
|---|---|---|
| **FastAPI** | Async-native; auto-generated OpenAPI docs; Pydantic validation; lifespan hooks for background tasks | Slightly more setup than Flask |
| Flask | Simple, well-known | Sync by default; background tasks require threads or Celery |
| Django | Batteries-included ORM and admin | Heavy for a single-service app; admin UI doesn't fit labeling UX |

### Vision Model

| Option | Pros | Cons |
|---|---|---|
| **YOLOv8 (ultralytics)** | Detects cats + humans in one pass; runs on CPU; COCO-pretrained (no initial dataset needed); easy fine-tuning API | Larger dependency; overkill if only cats matter |
| OpenCV Haar cascades | Tiny, no dependencies | Cat cascade is poor quality; no human detection without a separate model |
| Claude Vision API | Nuanced scene understanding; no local model | Latency (~1-3 s/frame); per-call cost; requires internet; can't fine-tune |
| MobileNet + custom head | Lightweight | Requires a pre-labeled dataset before the service can run at all |

### Persistence

| Option | Pros | Cons |
|---|---|---|
| **SQLite + SQLAlchemy** | Zero-config; file-based; survives reboots; sufficient for thousands of detections | Not suitable if multiple write processes needed; no built-in replication |
| PostgreSQL | Production-grade | Requires a running server; overkill for a home service |
| JSON flat files | No dependencies | Querying is painful; no atomic writes |

---

## Decision

- **Language:** Python 3.11+
- **Web framework:** FastAPI with uvicorn
- **Vision model:** YOLOv8 via `ultralytics` (default: `yolov8n.pt` for speed; swappable via config)
- **Persistence:** SQLite via SQLAlchemy 2.0 ORM (async-compatible session factory)
- **HTTP client:** `httpx` (async, consistent with FastAPI's async model)
- **Configuration:** `pydantic-settings` reading from `.env`

YOLOv8 is the pivotal choice: it detects both cats (class 15) and humans (class 0) in a single inference pass, meaning the human-guard rule costs nothing extra. The `ultralytics` library also exposes a `model.train()` API that maps directly onto the fine-tuning user story.

---

## Consequences

**Positive:**
- No separate model for human detection — one inference call covers both detection rules
- `ultralytics` fine-tuning API means US-6 (fine-tuning trigger) requires only dataset preparation, not model training infrastructure
- SQLite requires zero ops — the database is a file next to the service
- FastAPI lifespan context manager gives a clean hook to start/stop the camera loop alongside the HTTP server

**Negative:**
- Python's GIL means the camera loop and request handlers share one OS thread pool; heavy inference could add latency to API responses. Mitigated by running inference in a thread pool executor (`run_in_executor`)
- `yolov8n.pt` is ~6 MB and downloaded on first run — requires internet access at startup if the file is absent
- SQLite's single-writer constraint means the camera loop and API writes could contend under high frame rates. Not an issue at 5-second intervals; revisit if interval drops below ~1 second

**Future options unlocked:**
- Swapping to a Claude Vision API backend requires only replacing `detector.py:detect()` — the rest of the pipeline is model-agnostic
- Migrating to PostgreSQL requires only changing `DATABASE_URL` — SQLAlchemy abstracts the driver
