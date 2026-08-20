from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


CHANNEL_ORDER = ("CH1-1", "CH1-2", "CH2-1", "CH2-2")


@dataclass(frozen=True)
class VideoFrame:
    bgr: Any
    fps: float
    frame_count: int


@dataclass
class LoadedAnalysisProject:
    wet_area_polygons: dict[str, list[list[tuple[float, float]]]]
    manual_exclusions: dict[str, list[dict[str, Any]]]


def read_video_frame(video_path: Path, frame_index: int) -> VideoFrame:
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_index < 0 or (frame_count and frame_index >= frame_count):
        capture.release()
        raise ValueError(f"Frame {frame_index} is outside video range")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return VideoFrame(frame, fps, frame_count)


def load_project(project_path: Path, frame_shape: tuple[int, int]) -> dict[str, Any]:
    project = json.loads(project_path.read_text(encoding="utf-8"))
    requested = project.get("channels", list(CHANNEL_ORDER))
    if not isinstance(requested, list) or not requested:
        raise ValueError("Project must select at least one result channel")
    unknown = [channel for channel in requested if channel not in CHANNEL_ORDER]
    if unknown:
        raise ValueError(f"Unknown result channels: {', '.join(map(str, unknown))}")
    channels = [channel for channel in CHANNEL_ORDER if channel in requested]
    project["channels"] = channels
    for channel in channels:
        if len(project.get("rois", {}).get(channel, [])) < 3:
            raise ValueError(f"Incomplete ROI: {channel}")
        if len(project.get("zones", {}).get(channel, [])) < 3:
            raise ValueError(f"Incomplete interest zone: {channel}")
    height, width = frame_shape
    source = project.get("source", {})
    if (int(source.get("frame_width", width)), int(source.get("frame_height", height))) != (width, height):
        raise ValueError("Project dimensions do not match the video frame")
    return project


def load_analysis_project(
    analysis_path: Path,
    roi_project_path: Path,
    roi_sha256: str,
    frame_shape: tuple[int, int],
    frame_index: int,
) -> LoadedAnalysisProject:
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported analysis project schema")
    if str(payload.get("roi_project", {}).get("sha256", "")) != roi_sha256:
        raise RuntimeError("Generated ROI profile changed; regenerate analysis files")
    source = payload.get("source", {})
    height, width = frame_shape
    if (int(source.get("frame_width", width)), int(source.get("frame_height", height))) != (width, height):
        raise RuntimeError("Analysis dimensions do not match frame")
    if int(source.get("reference_frame", frame_index)) != frame_index:
        raise RuntimeError("Analysis reference frame differs from requested frame")
    wet: dict[str, list[list[tuple[float, float]]]] = {}
    exclusions: dict[str, list[dict[str, Any]]] = {}
    raw_frame = payload.get("frame_exclusions", {}).get(str(frame_index), {})
    for channel in CHANNEL_ORDER:
        wet[channel] = [
            [(float(x), float(y)) for x, y in polygon]
            for polygon in payload.get("wet_area_polygons", {}).get(channel, [])
        ]
        exclusions[channel] = []
        for item in raw_frame.get(channel, []):
            points = [(float(x), float(y)) for x, y in item.get("points", [])]
            if len(points) < 3:
                raise ValueError(f"Invalid exclusion {item.get('id')} in {channel}")
            exclusions[channel].append({"id": str(item["id"]), "kind": str(item["kind"]), "points": points})
    return LoadedAnalysisProject(wet, exclusions)
