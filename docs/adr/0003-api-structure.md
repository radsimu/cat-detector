# ADR-0003: API Structure

**Status:** Accepted  
**Date:** 2026-04-06  
**Author:** Cat Detector Team

---

## Context

The service exposes both a user-facing web UI and a machine-readable API. We need to decide how to structure those endpoints, whether the UI should be server-rendered or a separate frontend, and how the fine-tuning trigger should be handled given it is a long-running background operation.

Three specific questions drove this ADR:

1. **Server-rendered vs. SPA frontend** — do we ship a separate JS bundle, or render HTML server-side?
2. **API versioning** — do we namespace under `/api/v1/...` from the start?
3. **Async operations** — the fine-tuning trigger is long-running; how do we model that in the API?

---

## Options Considered

### Question 1: UI rendering approach

| Option | Pros | Cons |
|---|---|---|
| **Server-rendered Jinja2 + vanilla JS for dynamic parts** | No build tooling; single Python process; no CORS; trivially deployable | Less interactive than a SPA; refreshing data requires fetch calls |
| React/Vue SPA (separate) | Rich interactivity; separates concerns | Requires Node.js, a build step, CORS config, and separate deployment |
| HTMX | Minimal JS; HTML-driven partial updates | Another dependency; team may be unfamiliar |

**Decision:** Jinja2 template for the shell, vanilla `fetch()` for dynamic data loading. The UI is simple enough (a card grid + two buttons per card) that a framework adds no value.

### Question 2: API versioning

| Option | Pros | Cons |
|---|---|---|
| `/api/v1/...` from day one | Future-proof; clean breaking-change story | Premature for a project with one consumer (its own UI) |
| **`/api/...` without version prefix** | Simpler URLs; easy to add versioning later | Harder to introduce breaking changes without coordination |
| No `/api` prefix (resources at root) | Shortest URLs | Ambiguous — hard to tell which routes serve UI vs. data |

**Decision:** `/api/` prefix, no version. The only consumer of this API is the bundled UI. A version prefix would add ceremony with no benefit at this scope. If the API is ever exposed externally, add `/api/v1/` at that point via a router prefix change.

### Question 3: Long-running fine-tuning trigger

| Option | Pros | Cons |
|---|---|---|
| **Synchronous POST returns immediately; status polled via GET** | Simple; no WebSocket or SSE infrastructure | Client must poll; no push notification when done |
| WebSocket / SSE for progress streaming | Real-time feedback | Adds meaningful complexity for a training project |
| Celery task queue | Production-grade; retries, monitoring | Requires Redis/RabbitMQ; severe overkill here |
| `BackgroundTasks` (FastAPI built-in) | Zero extra infrastructure; runs in-process | Not durable — a crash loses the task; only one worker |

**Decision:** `BackgroundTasks` for the trigger (POST `/api/training/trigger`), with a polling endpoint (GET `/api/training/status`). The UI polls every 10 seconds. This is explicitly a training-project trade-off — a crash during fine-tuning is acceptable; durability is not required.

---

## Decision

### Route map

```
GET  /                              HTML shell (Jinja2)

GET  /api/detections                List detections (paginated: ?skip=&limit=)
POST /api/detections/{id}/label     Update label for a detection
GET  /api/detections/{id}/frame     Serve annotated frame image (JPEG)

GET  /api/training/status           Training state + label counts
POST /api/training/trigger          Start fine-tuning in background

GET  /static/*                      CSS, JS, images (StaticFiles mount)
```

### Design rules applied

- **Nouns, not verbs** in resource paths (`/detections`, not `/getDetections`)
- **POST for actions** that are not pure CRUD (`/label`, `/trigger`) — these mutate state but don't map cleanly to PUT/PATCH semantics
- **No nested resources beyond one level** — `/detections/{id}/label` is the deepest nesting
- **Pagination via query params** (`skip` / `limit`) — cursor-based pagination is unnecessary at this data volume
- **Frame served by the API, not directly from disk** — keeps the frame path an internal implementation detail; the API can add auth headers, cache headers, or redirect to object storage later without changing the UI

---

## Consequences

**Positive:**
- No build tooling required — `pip install` and run
- Auto-generated OpenAPI docs at `/docs` (FastAPI default) document all endpoints for free
- The `/frame` endpoint abstracts storage location — migrating to S3 later requires only changing that one endpoint
- Polling for training status is trivially implemented with `setInterval` in the UI

**Negative:**
- The UI is tightly coupled to this API — any breaking change to response shapes breaks the UI. Acceptable because they are co-deployed and co-versioned
- `BackgroundTasks` runs in the same process as the HTTP server. A long fine-tuning run will consume CPU alongside live inference. Mitigated by running fine-tuning only on demand, not continuously
- No OpenAPI client generation is set up — the UI uses raw `fetch()` calls. Type drift between API and UI is possible. Acceptable for this scope; add `openapi-ts` code-gen if the API surface grows

**Explicitly out of scope:**
- Rate limiting
- Response caching (ETags, Cache-Control)
- Bulk label update endpoint
- Streaming fine-tuning progress
