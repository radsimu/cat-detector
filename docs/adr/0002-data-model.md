# ADR-0002: Data Model Design

**Status:** Accepted  
**Date:** 2026-04-06  
**Author:** Cat Detector Team

---

## Context

We need to persist information about each detection event so that:

1. The web UI can display detection history with frames and metadata
2. Users can label events as true or false positive
3. The fine-tuning pipeline can query labeled frames
4. We can answer basic questions: "how often are cats detected?", "what's the false positive rate?"

The data model must balance simplicity (training-project scope, one developer) against being genuinely useful as a training dataset for the fine-tuning use case.

Three related design questions emerged:

- **What constitutes a "detection event"?** Every sampled frame, or only frames where something was found?
- **Where do frame images live?** In the database (as BLOBs) or on disk?
- **How is labeling modelled?** Separate table or inline on the detection row?

---

## Options Considered

### Question 1: What to store as a detection event

| Option | Pros | Cons |
|---|---|---|
| **Store only frames with at least one cat or human** | Keeps the DB small; every row is potentially interesting | Loses "quiet" frames needed for true-negative training examples |
| Store every sampled frame | Complete audit trail; better dataset for training | Floods the DB at 5-second intervals; ~17,000 rows/day |
| Store every frame, prune old "empty" ones | Best of both | Complex; requires a scheduled cleanup job |

**Decision:** Store only frames where `cat_count > 0 OR human_count > 0`. Empty frames are discarded. For true-negative examples, users can manually add images to `tests/frames/`.

### Question 2: Frame image storage

| Option | Pros | Cons |
|---|---|---|
| **Disk (filesystem), path stored in DB** | Fast reads; easy to inspect/delete; no DB size concerns | Frame and DB row can get out of sync if files are deleted manually |
| SQLite BLOB | Everything in one file; simpler backup | SQLite performance degrades with large BLOBs; difficult to inspect images |
| Object storage (S3, MinIO) | Scales; accessible from anywhere | Requires a running service; overkill for a home project |

**Decision:** Frames saved to a configurable `FRAMES_DIR` on disk. The DB stores the relative path. A missing file is handled gracefully by the API (404 on the `/frame` endpoint).

### Question 3: Labeling model

| Option | Pros | Cons |
|---|---|---|
| **Inline on Detection row (`label`, `labeled_at`)** | No join needed; simple queries for training data export | Can't store multiple labels per event (e.g. disagreement between labelers) |
| Separate `Label` table (FK to Detection) | Supports multiple labelers; audit trail of label changes | Over-engineered for a single-user home project |

**Decision:** Inline label on the `Detection` row. A three-value enum (`unlabeled`, `true_positive`, `false_positive`) covers all use cases. A `labeled_at` timestamp is included for future audit needs.

---

## Decision

Single `detections` table:

```
detections
├── id                    INTEGER PK
├── created_at            DATETIME (indexed)
├── cat_count             INTEGER
├── human_count           INTEGER
├── max_cat_confidence    FLOAT        -- highest cat confidence score in the frame
├── webhook_triggered     BOOLEAN
├── webhook_skipped_reason VARCHAR     -- null | "no_cats" | "human_present" | "cooldown"
├── frame_path            VARCHAR      -- relative path on disk, nullable
├── label                 VARCHAR      -- "unlabeled" | "true_positive" | "false_positive"
└── labeled_at            DATETIME     -- nullable
```

No foreign keys, no joins, no migrations tooling for v0 — SQLAlchemy's `create_all()` creates the schema on first run. Alembic is included as a dependency for when schema evolution becomes necessary.

`max_cat_confidence` stores only the highest per-frame confidence rather than individual bounding boxes. This is sufficient for labeling and UI display; bounding boxes are visible in the annotated frame image on disk.

---

## Consequences

**Positive:**
- All information needed to render the UI and export training data is in a single table — no joins
- `label` + `frame_path` is the entire interface between the labeling UI and the fine-tuning pipeline
- Schema is trivially readable by anyone with a SQLite browser
- `webhook_skipped_reason` makes it easy to audit why alerts were or weren't sent without parsing logs

**Negative:**
- Bounding box coordinates are not stored in the DB; if the annotated frame file is deleted, the spatial information is lost. Mitigate later by adding a `detections_boxes` child table if needed
- `max_cat_confidence` loses per-box granularity (e.g. two cats at 0.9 and 0.6 looks the same as one cat at 0.9). Acceptable for the labeling UX; revisit for training data quality analysis
- No schema migration tooling active in v0 — any column addition requires a manual `ALTER TABLE` or a drop-and-recreate of the SQLite file

**Schema evolution path:**
- Add Alembic migrations before any production deployment
- If multi-labeler support is needed, extract `label` and `labeled_at` into a `detection_labels` table with a `labeler_id` FK
