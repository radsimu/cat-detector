import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, computed_field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from cat_detector.camera import _grab_frame
from cat_detector.config import settings
from cat_detector.database import get_db
from cat_detector.detector import detect, save_frame
from cat_detector.models import Detection, Label

router = APIRouter(prefix="/api/detections", tags=["detections"])


class DetectionOut(BaseModel):
    id: int
    created_at: datetime
    cat_count: int
    dog_count: int
    human_count: int
    max_cat_confidence: float
    webhook_triggered: bool
    webhook_skipped_reason: str | None
    detections_json: str | None = None
    frame_path: str | None
    user_bbox: str | None = None
    user_bbox_class: str | None = None
    label: str

    model_config = {"from_attributes": True, "ser_json_inf_nan": "constants"}

    @computed_field
    @property
    def boxes(self) -> list[dict[str, Any]]:
        if self.detections_json:
            try:
                return json.loads(self.detections_json)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    def model_post_init(self, __context: Any) -> None:
        pass  # detections_json kept for ORM mapping but excluded below

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude", set())
        if isinstance(kwargs["exclude"], set):
            kwargs["exclude"].add("detections_json")
        return super().model_dump(**kwargs)


class LabelIn(BaseModel):
    label: Label


class BBoxIn(BaseModel):
    """User-drawn bounding box for a false_negative annotation."""
    bbox: list[float]  # [x1, y1, x2, y2] in frame pixel coordinates
    class_name: str = "cat"  # 'cat' or 'person'


_VALID_CLASSES = {"cat", "person"}


def _validate_bbox(bbox: list[float]) -> str:
    if len(bbox) != 4:
        raise HTTPException(status_code=422, detail="bbox must have 4 values [x1, y1, x2, y2]")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=422, detail="bbox must have x2>x1 and y2>y1")
    return json.dumps([round(float(v), 1) for v in bbox])


@router.get("", response_model=list[DetectionOut])
def list_detections(
    skip: int = 0,
    limit: int = 50,
    date_from: str | None = None,
    date_to: str | None = None,
    label_filter: str = "unlabeled",  # "unlabeled" | "labeled" | "all"
    db: Session = Depends(get_db),
):
    query = db.query(Detection)
    if date_from:
        query = query.filter(Detection.created_at >= date_from)
    if date_to:
        query = query.filter(Detection.created_at <= date_to + " 23:59:59")
    if label_filter == "unlabeled":
        query = query.filter(Detection.label == Label.unlabeled.value)
    elif label_filter == "labeled":
        query = query.filter(Detection.label != Label.unlabeled.value)
    # "all" → no label filter
    return (
        query
        .order_by(desc(Detection.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.post("/{detection_id}/label", response_model=DetectionOut)
def label_detection(
    detection_id: int,
    body: LabelIn,
    db: Session = Depends(get_db),
):
    detection = db.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    detection.label = body.label
    detection.labeled_at = datetime.utcnow()
    db.commit()
    db.refresh(detection)
    return detection


@router.post("/{detection_id}/annotate", response_model=DetectionOut)
def annotate_false_negative(
    detection_id: int,
    body: BBoxIn,
    db: Session = Depends(get_db),
):
    """Mark a detection as false_negative with a user-drawn bbox of the missed
    subject. class_name in the body specifies 'cat' or 'person'."""
    if body.class_name not in _VALID_CLASSES:
        raise HTTPException(status_code=422, detail=f"class_name must be one of {_VALID_CLASSES}")
    detection = db.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    detection.user_bbox = _validate_bbox(body.bbox)
    detection.user_bbox_class = body.class_name
    detection.label = Label.false_negative
    detection.labeled_at = datetime.utcnow()
    db.commit()
    db.refresh(detection)
    return detection


@router.delete("/{detection_id}")
def delete_detection(detection_id: int, db: Session = Depends(get_db)):
    detection = db.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    if detection.frame_path:
        try:
            p = Path(detection.frame_path)
            if p.exists():
                p.unlink()
        except OSError:
            pass  # delete the DB row even if the file is gone
    db.delete(detection)
    db.commit()
    return {"status": "deleted"}


@router.get("/{detection_id}/frame")
def get_frame(detection_id: int, db: Session = Depends(get_db)):
    detection = db.get(Detection, detection_id)
    if not detection or not detection.frame_path:
        raise HTTPException(status_code=404, detail="Frame not found")
    path = Path(detection.frame_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame file missing from disk")
    return FileResponse(str(path), media_type="image/jpeg")


@router.post("/report-missed", response_model=DetectionOut)
async def report_missed_cat(db: Session = Depends(get_db)):
    """
    Snapshot the current camera frame and save it as an unlabeled detection.
    The client is expected to follow up with POST /{id}/annotate to supply
    the user-drawn bbox and set the false_negative label.
    """
    frame = await asyncio.get_event_loop().run_in_executor(None, _grab_frame)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera unavailable")

    # Save the RAW frame (not annotated) — annotation will be drawn client-side
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_missed_cat.jpg"
    frame_path = save_frame(frame, settings.frames_dir, filename)

    # Run detection so we record what the model *did* see on this frame
    result = await asyncio.get_event_loop().run_in_executor(None, detect, frame)

    detection = Detection(
        cat_count=result.cat_count,
        dog_count=result.dog_count,
        human_count=result.human_count,
        max_cat_confidence=result.max_cat_confidence,
        webhook_triggered=False,
        webhook_skipped_reason="manual_report",
        frame_path=str(frame_path),
        label=Label.unlabeled,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


@router.post("/upload", response_model=DetectionOut)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload an arbitrary image and create an unlabeled detection from it.
    The client is expected to follow up with POST /{id}/annotate to supply a
    user-drawn bbox and class, the same flow as a live missed-cat report."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty file")

    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in (file.filename or "upload") if c.isalnum() or c in "._-")[:40]
    filename = f"{timestamp}_upload_{safe_name or 'image'}.jpg"
    frame_path = save_frame(frame, settings.frames_dir, filename)

    # Run detection to capture the model baseline on this image
    result = await asyncio.get_event_loop().run_in_executor(None, detect, frame)

    detection = Detection(
        cat_count=result.cat_count,
        dog_count=result.dog_count,
        human_count=result.human_count,
        max_cat_confidence=result.max_cat_confidence,
        detections_json=json.dumps(result.boxes) if result.boxes else None,
        webhook_triggered=False,
        webhook_skipped_reason="uploaded",
        frame_path=str(frame_path),
        label=Label.unlabeled,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection
