"""
API for sprinkler-event video recordings.
"""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from cat_detector.database import get_db
from cat_detector.models import Recording

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


class RecordingOut(BaseModel):
    id: int
    created_at: datetime
    detection_id: int | None
    file_path: str
    duration_seconds: float
    file_size_bytes: int

    model_config = {"from_attributes": True}


@router.get("", response_model=list[RecordingOut])
def list_recordings(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    return (
        db.query(Recording)
        .order_by(desc(Recording.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{recording_id}/video")
def get_video(recording_id: int, db: Session = Depends(get_db)):
    recording = db.get(Recording, recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    path = Path(recording.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file missing from disk")
    return FileResponse(str(path), media_type="video/mp4")


@router.delete("/{recording_id}")
def delete_recording(recording_id: int, db: Session = Depends(get_db)):
    recording = db.get(Recording, recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    try:
        path = Path(recording.file_path)
        if path.exists():
            path.unlink()
    except OSError:
        pass  # delete DB row anyway
    db.delete(recording)
    db.commit()
    return {"status": "deleted"}
