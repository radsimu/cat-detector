"""
Training / fine-tuning endpoint.

The actual fine-tuning implementation is a stub — replace with your chosen
training strategy (e.g. YOLOv8 fine-tune on labeled true/false positives,
or few-shot prompting adjustments for a vision LLM backend).
"""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from cat_detector.database import get_db
from cat_detector.models import Detection, Label

router = APIRouter(prefix="/api/training", tags=["training"])
logger = logging.getLogger(__name__)

_training_in_progress = False


class TrainingStatus(BaseModel):
    status: str
    true_positives: int
    false_positives: int
    false_negatives: int
    unlabeled: int


def _run_finetuning(true_positives: list[str], false_positives: list[str]) -> None:
    """
    Stub: implement your fine-tuning logic here.

    Suggested approach:
    - Export labeled frames to a YOLO dataset layout (images/labels dirs)
    - Call `model.train(data="dataset.yaml", epochs=10, imgsz=640)`
    - Save the new weights and update settings.detection_model
    """
    global _training_in_progress
    logger.info(
        "Fine-tuning started — %d true positives, %d false positives",
        len(true_positives),
        len(false_positives),
    )
    # TODO: implement actual training
    import time
    time.sleep(2)  # simulate work
    logger.info("Fine-tuning complete (stub)")
    _training_in_progress = False


@router.get("/status", response_model=TrainingStatus)
def training_status(db: Session = Depends(get_db)):
    counts: dict[Label, int] = {label: 0 for label in Label}
    for detection in db.query(Detection).all():
        try:
            counts[Label(detection.label)] += 1
        except ValueError:
            pass  # unknown legacy label — skip

    return TrainingStatus(
        status="in_progress" if _training_in_progress else "idle",
        true_positives=counts[Label.true_positive],
        false_positives=counts[Label.false_positive],
        false_negatives=counts[Label.false_negative],
        unlabeled=counts[Label.unlabeled],
    )


@router.post("/trigger")
def trigger_training(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    global _training_in_progress
    if _training_in_progress:
        return {"status": "already_running"}

    true_pos = [
        d.frame_path
        for d in db.query(Detection).filter(Detection.label == Label.true_positive).all()
        if d.frame_path
    ]
    false_neg = [
        d.frame_path
        for d in db.query(Detection).filter(Detection.label == Label.false_negative).all()
        if d.frame_path
    ]
    false_pos = [
        d.frame_path
        for d in db.query(Detection).filter(Detection.label == Label.false_positive).all()
        if d.frame_path
    ]

    # False negatives are positive examples (cat was there, model missed it)
    positive_examples = true_pos + false_neg

    if not positive_examples and not false_pos:
        return {"status": "no_labeled_data"}

    _training_in_progress = True
    background_tasks.add_task(_run_finetuning, positive_examples, false_pos)
    return {
        "status": "started",
        "true_positives": len(true_pos),
        "false_negatives": len(false_neg),
        "false_positives": len(false_pos),
    }
