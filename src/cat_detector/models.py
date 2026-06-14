from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from cat_detector.database import Base


class Label(str, Enum):
    unlabeled = "unlabeled"
    true_positive = "true_positive"
    false_positive = "false_positive"
    false_negative = "false_negative"


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    # What was detected
    cat_count: Mapped[int] = mapped_column(Integer, default=0)
    dog_count: Mapped[int] = mapped_column(Integer, default=0)
    human_count: Mapped[int] = mapped_column(Integer, default=0)
    max_cat_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Action taken
    webhook_triggered: Mapped[bool] = mapped_column(default=False)
    webhook_skipped_reason: Mapped[str | None] = mapped_column(String, nullable=True)

    # Per-box detection details: JSON array of {class_id, class_name, confidence, bbox}.
    # Stored as Text + json.loads on read. Typical payload is <1KB (5-10 boxes).
    # If list queries become slow at high volume (>10k rows), consider migrating
    # to a proper JSON column (PostgreSQL) or a related BoundingBox table.
    detections_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Frame stored on disk (relative path under frames_dir)
    frame_path: Mapped[str | None] = mapped_column(String, nullable=True)

    # User-drawn bounding box for false_negative annotations, JSON: [x1, y1, x2, y2]
    # Coordinates are in the frame's pixel space. Used by the fine-tuning pipeline.
    user_bbox: Mapped[str | None] = mapped_column(String, nullable=True)
    # Class the user annotated ('cat' or 'person'). Null for legacy records —
    # defaults to 'cat' in that case for backward compatibility.
    user_bbox_class: Mapped[str | None] = mapped_column(String, nullable=True)

    # Human labeling
    label: Mapped[str] = mapped_column(String, default=Label.unlabeled)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Recording(Base):
    """Video clip captured around a webhook-triggering detection."""
    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    detection_id: Mapped[int | None] = mapped_column(
        ForeignKey("detections.id"), nullable=True, index=True
    )
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
