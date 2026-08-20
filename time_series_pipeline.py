from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fluid_io import load_project
from phase_analysis_core import (
    CalibrationResult,
    build_static_exclusion_masks,
    calculate_phase,
    calibrate_marked_swatch,
    compose_channel_masks,
)


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(str(video_path), backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    return capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLUID-Space time-series pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--frame-step", type=int, required=True)
    parser.add_argument("--cad-roi-masks", required=True)
    parser.add_argument("--cad-placement-config")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_metadata(video_path: Path) -> tuple[float, int, tuple[int, int]]:
    capture = open_video_capture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError("Video FPS is unavailable or invalid")
    if frame_count <= 0:
        raise ValueError("Video frame count is unavailable or invalid")
    return fps, frame_count, (height, width)


def sampled_frame_indices(start: int, end: int, step: int, frame_count: int) -> list[int]:
    if start < 0:
        raise ValueError("Start frame cannot be negative")
    if end < start:
        raise ValueError("End frame must be greater than or equal to start frame")
    if step < 1:
        raise ValueError("Frame step must be at least 1")
    if end >= frame_count:
        raise ValueError(f"End frame {end} is outside video range (last frame: {frame_count - 1})")
    return list(range(start, end + 1, step))


def iter_video_frames(
    video_path: Path,
    frame_indices: list[int],
    expected_shape: tuple[int, int],
) -> Iterator[tuple[int, np.ndarray]]:
    capture = open_video_capture(video_path)
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot read frame {frame_index} from {video_path.name}")
            if frame.shape[:2] != expected_shape:
                raise ValueError(
                    f"Frame {frame_index} is {frame.shape[1]}x{frame.shape[0]}; expected "
                    f"{expected_shape[1]}x{expected_shape[0]}"
                )
            yield frame_index, frame
    finally:
        capture.release()


def circular_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty calibration series")
    angles = np.asarray(values, dtype=np.float64) * (2.0 * np.pi)
    vector = np.mean(np.exp(1j * angles))
    if abs(vector) < 1e-12:
        raise ValueError("Calibration hue shifts have no stable circular mean")
    return float((np.angle(vector) / (2.0 * np.pi) + 0.5) % 1.0 - 0.5)


def load_cad_masks(
    path: Path, shape: tuple[int, int], channels: tuple[str, ...]
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        masks = {channel: np.asarray(payload[channel]).astype(bool) for channel in channels}
    for channel, mask in masks.items():
        if mask.shape != shape:
            raise ValueError(f"CAD mask shape differs for {channel}: {mask.shape}, expected {shape}")
        if not mask.any():
            raise ValueError(f"CAD mask is empty for {channel}")
    return masks


def build_time_series_masks(
    shape: tuple[int, int],
    roi_project: dict[str, Any],
    cad_masks: dict[str, np.ndarray],
    channels: tuple[str, ...],
) -> dict[str, Any]:
    calibration_mask, _ = build_static_exclusion_masks(shape, roi_project)
    empty = np.zeros(shape, dtype=bool)
    return {
        channel: compose_channel_masks(
            channel,
            cad_masks[channel],
            cad_masks[channel],
            cad_masks[channel],
            calibration_mask,
            empty,
            empty,
            empty,
        )
        for channel in channels
    }


def next_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    number = 1
    while (output_root / f"run_{number:03d}").exists():
        number += 1
    run_dir = output_root / f"run_{number:03d}"
    run_dir.mkdir()
    return run_dir


def save_time_series_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "frame_index",
        "video_time_sec",
        "elapsed_time_sec",
        "channel",
        "mean_phase_index",
        "std_phase_index",
        "valid_pixels",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_time_series_graph(
    path: Path, rows: list[dict[str, Any]], channels: tuple[str, ...]
) -> None:
    column_count = 1 if len(channels) == 1 else 2
    row_count = (len(channels) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.2 * column_count, 3.7 * row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, channel in zip(axes.flat, channels):
        channel_rows = [row for row in rows if row["channel"] == channel]
        elapsed = [float(row["elapsed_time_sec"]) for row in channel_rows]
        means = [float(row["mean_phase_index"]) for row in channel_rows]
        axis.plot(elapsed, means, color="#1976a3", linewidth=1.6, marker="o", markersize=2.5)
        axis.axhline(1.5, color="#222222", linestyle="--", linewidth=0.9)
        axis.set_title(channel)
        axis.set_ylim(1.0, 2.0)
        axis.set_ylabel("Mean phase index")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(channels) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Elapsed time from start frame (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path | None:
    video_path = Path(args.video).resolve()
    project_path = Path(args.project).resolve()
    cad_masks_path = Path(args.cad_roi_masks).resolve()
    output_root = Path(args.output_root).resolve()
    fps, frame_count, shape = video_metadata(video_path)
    indices = sampled_frame_indices(
        args.start_frame, args.end_frame, args.frame_step, frame_count
    )
    roi_project = load_project(project_path, shape)
    channels = tuple(roi_project["channels"])
    calibration_names = tuple(
        name
        for name, prefix in (("CAL-CH1", "CH1"), ("CAL-CH2", "CH2"))
        if any(channel.startswith(prefix) for channel in channels)
    )
    cad_masks = load_cad_masks(cad_masks_path, shape, channels)
    masks = build_time_series_masks(shape, roi_project, cad_masks, channels)

    calibration_by_frame: dict[int, dict[str, CalibrationResult]] = {}
    for frame_index, frame_bgr in iter_video_frames(video_path, indices, shape):
        try:
            calibration_by_frame[frame_index] = {
                name: calibrate_marked_swatch(frame_bgr, roi_project, name)
                for name in calibration_names
            }
        except Exception as exc:
            raise RuntimeError(f"Calibration failed at frame {frame_index}: {exc}") from exc

    averaged_calibrations: dict[str, CalibrationResult] = {}
    for name in calibration_names:
        values = [calibration_by_frame[index][name].hue_shift for index in indices]
        averaged_calibrations[name] = replace(
            calibration_by_frame[indices[0]][name], hue_shift=circular_mean(values)
        )

    rows: list[dict[str, Any]] = []
    for frame_index, frame_bgr in iter_video_frames(video_path, indices, shape):
        for channel in channels:
            calibration_name = "CAL-CH1" if channel.startswith("CH1") else "CAL-CH2"
            result = calculate_phase(
                frame_bgr, masks[channel], averaged_calibrations[calibration_name]
            )
            summary = result.summary()
            if summary["mean_index"] is None:
                raise RuntimeError(f"No valid phase pixels for {channel} at frame {frame_index}")
            rows.append(
                {
                    "frame_index": frame_index,
                    "video_time_sec": frame_index / fps,
                    "elapsed_time_sec": (frame_index - args.start_frame) / fps,
                    "channel": channel,
                    "mean_phase_index": summary["mean_index"],
                    "std_phase_index": summary["std_index"],
                    "valid_pixels": summary["valid_pixels"],
                }
            )

    if args.check:
        print(
            f"Time-series check passed: {len(indices)} frames, "
            f"{indices[0]}..{indices[-1]}, step {args.frame_step}"
        )
        return None

    run_dir = next_run_directory(output_root)
    graph_dir = run_dir / "graphs"
    graph_dir.mkdir()
    save_time_series_csv(run_dir / "time_series.csv", rows)
    save_time_series_graph(graph_dir / "phase_time_series.png", rows, channels)

    channel_summary: dict[str, Any] = {}
    for channel in channels:
        channel_rows = [row for row in rows if row["channel"] == channel]
        values = np.asarray([row["mean_phase_index"] for row in channel_rows], dtype=float)
        channel_summary[channel] = {
            "samples": len(channel_rows),
            "valid_pixels_per_frame": int(channel_rows[0]["valid_pixels"]),
            "mean_of_frame_means": float(values.mean()),
            "minimum_frame_mean": float(values.min()),
            "maximum_frame_mean": float(values.max()),
        }

    placement_path = (
        Path(args.cad_placement_config).resolve() if args.cad_placement_config else None
    )
    summary = {
        "pipeline": "FLUID-Space time series",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(video_path),
            "sha256": file_sha256(video_path),
            "fps": fps,
            "frame_count": frame_count,
        },
        "frame_range": {
            "start": args.start_frame,
            "end": args.end_frame,
            "step": args.frame_step,
            "sampled_frames": indices,
        },
        "result_channels": list(channels),
        "roi_project": {"path": str(project_path), "sha256": file_sha256(project_path)},
        "cad": {
            "masks_path": str(cad_masks_path),
            "masks_sha256": file_sha256(cad_masks_path),
            "placement_config": (
                {"path": str(placement_path), "sha256": file_sha256(placement_path)}
                if placement_path is not None
                else None
            ),
        },
        "geometry_rule": "CAD interior AND channel ROI AND channel Interest Zone, minus calibration ROI",
        "bubble_policy": "Annotations retained in profile; not used in time-series analysis",
        "calibration": {
            name: {
                "average_hue_shift": averaged_calibrations[name].hue_shift,
                "per_frame_hue_shift": [
                    {"frame_index": index, "hue_shift": calibration_by_frame[index][name].hue_shift}
                    for index in indices
                ],
            }
            for name in calibration_names
        },
        "channels": channel_summary,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"Time-series result: {run_dir}")
    return run_dir


def main() -> None:
    try:
        run(parse_args())
    except Exception as error:
        print(
            f"Time-series pipeline stopped ({type(error).__name__}): {error}",
            flush=True,
        )
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
