from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator


BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_TRACK_FRAMES = 5000
MODEL_DIR = BASE_DIR / "models"
DEFAULT_ACCIDENT_MODEL_PATH = BASE_DIR / "models" / "accident-yolo26n.pt"
app = FastAPI(title="Traffic Accident ROI Annotation API", version="1.0.0")
_accident_model = None
_accident_model_path: Path | None = None
_accident_model_lock = Lock()


class RoiAnnotation(BaseModel):
    label: str = Field(min_length=1, max_length=200)
    eventType: Literal[
        "collision", "near_miss", "stopped_vehicle", "wrong_way", "other"
    ] = "other"
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)
    width: float = Field(gt=0, allow_inf_nan=False)
    height: float = Field(gt=0, allow_inf_nan=False)
    frameStart: int = Field(ge=0)
    frameEnd: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_frame_range(self) -> "RoiAnnotation":
        if self.frameEnd < self.frameStart:
            raise ValueError("frameEnd must be greater than or equal to frameStart.")
        return self


class EvaluationRequest(BaseModel):
    predictions: list[RoiAnnotation]
    ground_truth: list[RoiAnnotation]
    iou_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


def _intersection_over_union(a: RoiAnnotation, b: RoiAnnotation) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _temporal_iou(a: RoiAnnotation, b: RoiAnnotation) -> float:
    intersection = max(
        0,
        min(a.frameEnd, b.frameEnd) - max(a.frameStart, b.frameStart) + 1,
    )
    union = max(a.frameEnd, b.frameEnd) - min(a.frameStart, b.frameStart) + 1
    return intersection / union if union > 0 else 0.0


def _spatiotemporal_iou(
    a: RoiAnnotation,
    b: RoiAnnotation,
) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    intersection_area = max(0.0, right - left) * max(0.0, bottom - top)
    overlap_frames = max(
        0,
        min(a.frameEnd, b.frameEnd) - max(a.frameStart, b.frameStart) + 1,
    )
    intersection_volume = intersection_area * overlap_frames
    a_volume = a.width * a.height * (a.frameEnd - a.frameStart + 1)
    b_volume = b.width * b.height * (b.frameEnd - b.frameStart + 1)
    union_volume = a_volume + b_volume - intersection_volume
    return intersection_volume / union_volume if union_volume > 0 else 0.0


def _center_drift(a: RoiAnnotation, b: RoiAnnotation) -> float:
    ax = a.x + a.width / 2
    ay = a.y + a.height / 2
    bx = b.x + b.width / 2
    by = b.y + b.height / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _same_event_type(a: RoiAnnotation, b: RoiAnnotation) -> bool:
    return a.eventType == b.eventType


def _available_accident_models() -> list[dict[str, str]]:
    models: dict[str, str] = {}
    model_root = MODEL_DIR.resolve()
    if model_root.is_dir():
        for model_path in sorted(MODEL_DIR.glob("*.pt")):
            if not model_path.is_file():
                continue
            resolved_path = model_path.resolve()
            try:
                resolved_path.relative_to(model_root)
            except ValueError:
                continue
            models[model_path.name] = model_path.name

    configured_path = Path(
        os.environ.get("ACCIDENT_MODEL_PATH", str(DEFAULT_ACCIDENT_MODEL_PATH))
    ).expanduser()
    if not configured_path.is_absolute():
        configured_path = BASE_DIR / configured_path
    configured_path = configured_path.resolve()
    if configured_path.is_file():
        try:
            configured_id = configured_path.relative_to(MODEL_DIR.resolve()).as_posix()
        except ValueError:
            configured_id = "__configured__"
        models[configured_id] = configured_path.name

    return [{"id": model_id, "name": name} for model_id, name in models.items()]


def _resolve_accident_model_path(model_id: str) -> Path:
    available = {model["id"] for model in _available_accident_models()}
    if model_id not in available:
        raise HTTPException(
            status_code=422,
            detail="Selected model is not available. Refresh the model list and choose an installed checkpoint.",
        )
    if model_id == "__configured__":
        configured_path = Path(
            os.environ.get("ACCIDENT_MODEL_PATH", str(DEFAULT_ACCIDENT_MODEL_PATH))
        ).expanduser()
        if not configured_path.is_absolute():
            configured_path = BASE_DIR / configured_path
        return configured_path.resolve()
    model_path = (MODEL_DIR / model_id).resolve()
    try:
        model_path.relative_to(MODEL_DIR.resolve())
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Invalid model selection.") from error
    return model_path


def _load_accident_model(model_id: str):
    global _accident_model, _accident_model_path

    model_path = _resolve_accident_model_path(model_id)
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model checkpoint '{model_path.name}' does not exist. "
                "Add a YOLO detection checkpoint to the models folder or configure ACCIDENT_MODEL_PATH."
            ),
        )

    with _accident_model_lock:
        if _accident_model is not None and _accident_model_path == model_path:
            return _accident_model
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise HTTPException(
                status_code=503,
                detail="Ultralytics is not installed; install the project requirements.",
            ) from error
        try:
            model = YOLO(str(model_path))
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load accident model checkpoint: {error}",
            ) from error
        if getattr(model, "task", None) not in (None, "detect"):
            raise HTTPException(
                status_code=422,
                detail="The configured accident checkpoint must be a YOLO detection model.",
            )
        _accident_model = model
        _accident_model_path = model_path
        return model


def _event_type_from_label(label: str) -> str:
    normalized = label.lower().replace("-", "_").replace(" ", "_")
    if "near" in normalized and "miss" in normalized:
        return "near_miss"
    if "stopped" in normalized or "stationary" in normalized:
        return "stopped_vehicle"
    if "wrong" in normalized and "way" in normalized:
        return "wrong_way"
    if any(word in normalized for word in ("accident", "collision", "crash")):
        return "collision"
    return "other"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "video_roi2.html")


@app.get("/healthz")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def list_accident_models() -> dict[str, list[dict[str, str]]]:
    return {"models": _available_accident_models()}


@app.post("/api/evaluate")
def evaluate_rois(request: EvaluationRequest) -> dict:
    adjacency: list[list[tuple[int, float, float, float]]] = [
        [] for _ in request.predictions
    ]
    for prediction_index, prediction in enumerate(request.predictions):
        for truth_index, truth in enumerate(request.ground_truth):
            if not _same_event_type(prediction, truth):
                continue
            spatial_iou = _intersection_over_union(prediction, truth)
            temporal_iou = _temporal_iou(prediction, truth)
            score = _spatiotemporal_iou(
                prediction,
                truth,
            )
            if score <= 0:
                continue
            if score >= request.iou_threshold:
                adjacency[prediction_index].append(
                    (truth_index, score, spatial_iou, temporal_iou)
                )
    for edges in adjacency:
        edges.sort(key=lambda edge: edge[1], reverse=True)

    truth_owner: dict[int, int] = {}
    prediction_matches: dict[int, tuple[int, float, float, float]] = {}

    def assign(prediction_index: int, visited_truths: set[int]) -> bool:
        for edge in adjacency[prediction_index]:
            truth_index = edge[0]
            if truth_index in visited_truths:
                continue
            visited_truths.add(truth_index)
            previous_owner = truth_owner.get(truth_index)
            if previous_owner is None or assign(previous_owner, visited_truths):
                truth_owner[truth_index] = prediction_index
                prediction_matches[prediction_index] = edge
                return True
        return False

    for prediction_index in range(len(request.predictions)):
        assign(prediction_index, set())

    matches = []
    for prediction_index, (truth_index, score, spatial_iou, temporal_iou) in sorted(
        prediction_matches.items()
    ):
        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": truth_index,
                "spatial_iou": spatial_iou,
                "temporal_iou": temporal_iou,
                "spatiotemporal_iou": score,
                "center_drift": _center_drift(
                    request.predictions[prediction_index],
                    request.ground_truth[truth_index],
                ),
            }
        )

    true_positives = len(matches)
    false_positives = len(request.predictions) - true_positives
    false_negatives = len(request.ground_truth) - true_positives
    precision = true_positives / len(request.predictions) if request.predictions else 0.0
    recall = true_positives / len(request.ground_truth) if request.ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_count = max(1, true_positives)

    return {
        "threshold": request.iou_threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_spatial_iou": sum(m["spatial_iou"] for m in matches) / matched_count,
        "mean_temporal_iou": sum(m["temporal_iou"] for m in matches) / matched_count,
        "mean_spatiotemporal_iou": sum(m["spatiotemporal_iou"] for m in matches)
        / matched_count,
        "mean_center_drift": sum(m["center_drift"] for m in matches) / matched_count,
        "matches": matches,
    }


@app.post("/api/auto-label")
def auto_label_video(
    video: UploadFile = File(...),
    model_id: str = Form(...),
    annotation_fps: float = Form(30),
    confidence_threshold: float = Form(0.25),
) -> dict:
    if not video.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")
    if (
        not math.isfinite(annotation_fps)
        or annotation_fps <= 0
        or annotation_fps > 120
        or not math.isfinite(confidence_threshold)
        or confidence_threshold < 0
        or confidence_threshold > 1
    ):
        raise HTTPException(
            status_code=422,
            detail="annotation_fps must be at most 120 and confidence_threshold must be between 0 and 1.",
        )

    model = _load_accident_model(model_id)
    suffix = Path(video.filename).suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".video"
    temp_path = None
    try:
        size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while chunk := video.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                temp_file.write(chunk)

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            capture.release()
            raise HTTPException(status_code=415, detail="Could not read the uploaded video.")
        try:
            video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = annotation_fps

            rois = []
            frame_number = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                try:
                    with _accident_model_lock:
                        results = model.predict(
                            source=frame,
                            conf=confidence_threshold,
                            verbose=False,
                        )
                except Exception as error:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Accident model inference failed on frame {frame_number}: {error}",
                    ) from error
                result = results[0]
                boxes = result.boxes
                if boxes is not None:
                    class_names = result.names
                    annotation_frame = round(frame_number / source_fps * annotation_fps)
                    for detection_index, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_id = int(box.cls[0].item())
                        label = str(class_names[class_id])
                        rois.append(
                            {
                                "id": f"auto-{frame_number}-{detection_index}",
                                "label": label,
                                "eventType": _event_type_from_label(label),
                                "x": round(float(x1), 2),
                                "y": round(float(y1), 2),
                                "width": round(float(x2 - x1), 2),
                                "height": round(float(y2 - y1), 2),
                                "frameStart": annotation_frame,
                                "frameEnd": annotation_frame,
                                "confidence": round(float(box.conf[0].item()), 4),
                            }
                        )
                frame_number += 1

            if frame_number == 0:
                raise HTTPException(status_code=422, detail="The uploaded video contains no readable frames.")
            return {
                "video_width": video_width,
                "video_height": video_height,
                "video_fps": source_fps,
                "processed_frames": frame_number,
                "rois": rois,
            }
        finally:
            capture.release()
    finally:
        video.file.close()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _create_tracker():
    tracker_factory = getattr(cv2, "TrackerCSRT_create", None)
    if tracker_factory is None and hasattr(cv2, "legacy"):
        tracker_factory = getattr(cv2.legacy, "TrackerCSRT_create", None)
    if tracker_factory is None:
        raise HTTPException(
            status_code=503,
            detail="CSRT tracker is unavailable; install opencv-contrib-python-headless.",
        )
    return tracker_factory()


@app.post("/api/track")
def track_roi(
    video: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
    start_frame: int = Form(...),
    frame_count: int = Form(300),
    annotation_fps: float = Form(30),
) -> dict:
    if not video.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")
    if (
        not all(math.isfinite(value) for value in (x, y, width, height, annotation_fps))
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
    ):
        raise HTTPException(
            status_code=422,
            detail="ROI coordinates and annotation FPS must be finite and positive.",
        )
    if (
        start_frame < 0
        or frame_count < 1
        or frame_count > MAX_TRACK_FRAMES
        or annotation_fps <= 0
        or annotation_fps > 120
    ):
        raise HTTPException(
            status_code=422,
            detail=f"frame_count must be between 1 and {MAX_TRACK_FRAMES}; annotation_fps must be at most 120.",
        )

    suffix = Path(video.filename).suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".video"
    temp_path = None
    try:
        size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while chunk := video.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                temp_file.write(chunk)

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise HTTPException(status_code=415, detail="Could not read the uploaded video.")
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = annotation_fps
            source_start_frame = round(start_frame / annotation_fps * source_fps)
            source_frame_count = max(
                1,
                math.ceil(frame_count / annotation_fps * source_fps),
            )
            if source_frame_count > MAX_TRACK_FRAMES:
                raise HTTPException(
                    status_code=422,
                    detail=f"Tracking span exceeds {MAX_TRACK_FRAMES} source frames; shorten the ROI frame range.",
                )
            if total_frames <= source_start_frame:
                raise HTTPException(
                    status_code=422,
                    detail="start_frame is outside the uploaded video.",
                )

            capture.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)
            ok, frame = capture.read()
            if not ok:
                raise HTTPException(status_code=422, detail="Could not read start_frame.")

            frame_height, frame_width = frame.shape[:2]
            if x + width > frame_width or y + height > frame_height:
                raise HTTPException(
                    status_code=422,
                    detail=f"ROI must fit within the video frame ({frame_width}x{frame_height}).",
                )

            tracker = _create_tracker()
            tracker_x = int(x)
            tracker_y = int(y)
            tracker_width = min(frame_width - tracker_x, max(1, int(width + 0.5)))
            tracker_height = min(frame_height - tracker_y, max(1, int(height + 0.5)))
            initialized = tracker.init(
                frame,
                (tracker_x, tracker_y, tracker_width, tracker_height),
            )
            if initialized is False:
                raise HTTPException(status_code=422, detail="CSRT could not initialize on this ROI.")
            boxes = [
                {
                    "frame": start_frame,
                    "time_seconds": round(source_start_frame / source_fps, 4),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(width, 2),
                    "height": round(height, 2),
                }
            ]
            tracked_frames = 1
            tracking_lost = False

            for frame_number in range(
                source_start_frame + 1,
                min(total_frames, source_start_frame + source_frame_count),
            ):
                ok, frame = capture.read()
                if not ok:
                    break
                tracking_ok, box = tracker.update(frame)
                if not tracking_ok:
                    tracking_lost = True
                    break
                box_x, box_y, box_width, box_height = box
                boxes.append(
                    {
                        "frame": round(frame_number / source_fps * annotation_fps),
                        "time_seconds": round(frame_number / source_fps, 4),
                        "x": round(float(box_x), 2),
                        "y": round(float(box_y), 2),
                        "width": round(float(box_width), 2),
                        "height": round(float(box_height), 2),
                    }
                )
                tracked_frames += 1

            return {
                "video_width": frame_width,
                "video_height": frame_height,
                "video_fps": source_fps,
                "tracked_frames": tracked_frames,
                "tracking_lost": tracking_lost,
                "boxes": boxes,
            }
        finally:
            capture.release()
    finally:
        video.file.close()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
