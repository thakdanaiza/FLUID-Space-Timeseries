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


PHASE_HISTOGRAM_EDGES = np.round(
    np.linspace(1.0, 2.0, 101, dtype=np.float64), 2
)
VIDEO_OUTPUT_KINDS = ("count", "percent")
VIDEO_FRAME_SIZE = (1600, 1000)


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
    parser.add_argument(
        "--video-output",
        action="append",
        choices=VIDEO_OUTPUT_KINDS,
        default=[],
        help="Histogram video type to export; repeat to select both types",
    )
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
        elapsed = np.asarray(
            [float(row["elapsed_time_sec"]) for row in channel_rows], dtype=float
        )
        means = np.asarray(
            [float(row["mean_phase_index"]) for row in channel_rows], dtype=float
        )
        standard_deviations = np.asarray(
            [float(row["std_phase_index"]) for row in channel_rows], dtype=float
        )
        axis.fill_between(
            elapsed,
            means - standard_deviations,
            means + standard_deviations,
            color="#1976a3",
            alpha=0.18,
            linewidth=0,
            label="Mean ± 1 SD",
        )
        axis.plot(
            elapsed,
            means,
            color="#1976a3",
            linewidth=1.6,
            marker="o",
            markersize=2.5,
            label="Mean",
        )
        axis.axhline(1.5, color="#222222", linestyle="--", linewidth=0.9)
        axis.set_title(channel)
        axis.set_ylim(1.0, 2.0)
        axis.set_ylabel("Mean phase index")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="best")
    for axis in axes.flat[len(channels) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Elapsed time from start frame (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def phase_histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    counts, _ = np.histogram(flattened, bins=PHASE_HISTOGRAM_EDGES)
    if int(counts.sum()) != int(flattened.size):
        raise ValueError("Phase histogram contains values outside the 1.0 to 2.0 range")
    percentages = counts.astype(np.float64) * (100.0 / max(1, int(counts.sum())))
    return counts.astype(np.int64), percentages


def save_phase_histogram_csv(
    path: Path,
    frame_indices: list[int],
    fps: float,
    start_frame: int,
    channels: tuple[str, ...],
    count_matrices: dict[str, np.ndarray],
    percent_matrices: dict[str, np.ndarray],
) -> None:
    fields = (
        "frame_index",
        "video_time_sec",
        "elapsed_time_sec",
        "channel",
        "bin_lower",
        "bin_upper",
        "bin_center",
        "pixel_count",
        "pixel_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for channel in channels:
            for frame_position, frame_index in enumerate(frame_indices):
                for bin_index in range(PHASE_HISTOGRAM_EDGES.size - 1):
                    lower = float(PHASE_HISTOGRAM_EDGES[bin_index])
                    upper = float(PHASE_HISTOGRAM_EDGES[bin_index + 1])
                    writer.writerow(
                        {
                            "frame_index": frame_index,
                            "video_time_sec": frame_index / fps,
                            "elapsed_time_sec": (frame_index - start_frame) / fps,
                            "channel": channel,
                            "bin_lower": lower,
                            "bin_upper": upper,
                            "bin_center": (lower + upper) / 2.0,
                            "pixel_count": int(count_matrices[channel][frame_position, bin_index]),
                            "pixel_percent": float(
                                percent_matrices[channel][frame_position, bin_index]
                            ),
                        }
                    )


def save_histogram_heatmaps(
    directory: Path,
    elapsed_times: np.ndarray,
    channels: tuple[str, ...],
    count_matrices: dict[str, np.ndarray],
    percent_matrices: dict[str, np.ndarray],
) -> dict[str, Any]:
    count_vmax = max(float(matrix.max()) for matrix in count_matrices.values())
    percent_vmax = max(float(matrix.max()) for matrix in percent_matrices.values())
    tick_indices = np.unique(
        np.linspace(0, elapsed_times.size - 1, min(8, elapsed_times.size), dtype=int)
    )
    files: dict[str, dict[str, str]] = {}
    configurations = (
        (
            "count",
            count_matrices,
            count_vmax,
            "Valid pixels per 0.01 phase bin",
        ),
        (
            "percent",
            percent_matrices,
            percent_vmax,
            "% valid pixels per 0.01 phase bin",
        ),
    )
    for channel in channels:
        files[channel] = {}
        for kind, matrices, vmax, colorbar_label in configurations:
            figure, axis = plt.subplots(figsize=(9.2, 5.2))
            image = axis.imshow(
                matrices[channel].T,
                origin="lower",
                aspect="auto",
                extent=(0.0, float(elapsed_times.size), 1.0, 2.0),
                cmap="magma",
                vmin=0.0,
                vmax=max(vmax, np.finfo(float).eps),
                interpolation="nearest",
            )
            axis.set_xticks(
                tick_indices.astype(float) + 0.5,
                [f"{elapsed_times[index]:.3g}" for index in tick_indices],
            )
            axis.set_xlabel("Elapsed time from start frame (s)")
            axis.set_ylabel("Phase index")
            axis.set_ylim(1.0, 2.0)
            axis.set_title(f"{channel} · Pixel {kind}")
            colorbar = figure.colorbar(image, ax=axis, pad=0.02)
            colorbar.set_label(colorbar_label)
            figure.tight_layout()
            filename = f"phase_histogram_{kind}_{channel}.png"
            figure.savefig(directory / filename, dpi=220)
            plt.close(figure)
            files[channel][kind] = f"histograms/{filename}"
    return {
        "count_color_limits": [0.0, count_vmax],
        "percent_color_limits": [0.0, percent_vmax],
        "files": files,
    }


def masked_frame_crop(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB tight crop with every pixel outside the analysis mask blacked out."""
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("Cannot create a video crop from an empty analysis mask")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    crop = cv2.cvtColor(frame_bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    crop_mask = mask[top:bottom, left:right]
    result = crop.copy()
    result[~crop_mask] = 0
    return result


def _histogram_video_figure(
    channel: str,
    kind: str,
    matrix: np.ndarray,
    vmax: float,
    elapsed_times: np.ndarray,
    crop_shape: tuple[int, int],
) -> tuple[Any, Any, Any, Any]:
    colorbar_label = (
        "Valid pixels per 0.01 phase bin"
        if kind == "count"
        else "% valid pixels per 0.01 phase bin"
    )
    tick_indices = np.unique(
        np.linspace(0, elapsed_times.size - 1, min(8, elapsed_times.size), dtype=int)
    )
    width, height = VIDEO_FRAME_SIZE
    figure = plt.figure(figsize=(width / 160.0, height / 160.0), dpi=160)
    grid = figure.add_gridspec(2, 1, height_ratios=(0.9, 1.1), hspace=0.34)
    video_axis = figure.add_subplot(grid[0])
    histogram_axis = figure.add_subplot(grid[1])

    source_image = video_axis.imshow(
        np.zeros((crop_shape[0], crop_shape[1], 3), dtype=np.uint8),
        interpolation="nearest",
    )
    video_axis.set_axis_off()
    video_title = video_axis.set_title(channel)

    heatmap = histogram_axis.imshow(
        matrix.T,
        origin="lower",
        aspect="auto",
        extent=(0.0, float(elapsed_times.size), 1.0, 2.0),
        cmap="magma",
        vmin=0.0,
        vmax=max(vmax, np.finfo(float).eps),
        interpolation="nearest",
    )
    histogram_axis.set_xticks(
        tick_indices.astype(float) + 0.5,
        [f"{elapsed_times[index]:.3g}" for index in tick_indices],
    )
    histogram_axis.set_xlabel("Elapsed time from start frame (s)")
    histogram_axis.set_ylabel("Phase index")
    histogram_axis.set_ylim(1.0, 2.0)
    histogram_axis.set_title(f"Pixel {kind}")
    time_line = histogram_axis.axvline(0.5, color="#ff2020", linewidth=2.2)
    colorbar = figure.colorbar(heatmap, ax=histogram_axis, pad=0.02)
    colorbar.set_label(colorbar_label)
    figure.subplots_adjust(left=0.08, right=0.91, top=0.95, bottom=0.09)
    return figure, source_image, video_title, time_line


def save_histogram_videos(
    directory: Path,
    video_path: Path,
    frame_indices: list[int],
    start_frame: int,
    source_fps: float,
    frame_step: int,
    channels: tuple[str, ...],
    output_kinds: tuple[str, ...],
    expected_shape: tuple[int, int],
    masks: dict[str, Any],
    count_matrices: dict[str, np.ndarray],
    percent_matrices: dict[str, np.ndarray],
) -> dict[str, Any]:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Histogram video output requires imageio-ffmpeg; install the project requirements"
        ) from exc

    output_fps = source_fps / frame_step
    matrices = {"count": count_matrices, "percent": percent_matrices}
    maxima = {
        "count": max(float(matrix.max()) for matrix in count_matrices.values()),
        "percent": max(float(matrix.max()) for matrix in percent_matrices.values()),
    }
    elapsed_times = np.asarray(
        [(index - start_frame) / source_fps for index in frame_indices], dtype=float
    )
    directory.mkdir()
    renderers: list[dict[str, Any]] = []
    files: dict[str, dict[str, str]] = {channel: {} for channel in channels}
    completed = False
    try:
        for channel in channels:
            final_mask = masks[channel].final_mask
            rows, columns = np.nonzero(final_mask)
            crop_shape = (
                int(rows.max() - rows.min() + 1),
                int(columns.max() - columns.min() + 1),
            )
            for kind in output_kinds:
                filename = f"phase_histogram_{kind}_{channel}.mp4"
                final_path = directory / filename
                temporary_path = directory / f".{filename}.tmp.mp4"
                figure, source_image, video_title, time_line = _histogram_video_figure(
                    channel,
                    kind,
                    matrices[kind][channel],
                    maxima[kind],
                    elapsed_times,
                    crop_shape,
                )
                writer = imageio_ffmpeg.write_frames(
                    str(temporary_path),
                    VIDEO_FRAME_SIZE,
                    fps=output_fps,
                    codec="libx264",
                    pix_fmt_in="rgb24",
                    pix_fmt_out="yuv420p",
                    macro_block_size=2,
                    ffmpeg_log_level="error",
                    output_params=["-movflags", "+faststart"],
                )
                try:
                    writer.send(None)
                except Exception:
                    writer.close()
                    plt.close(figure)
                    temporary_path.unlink(missing_ok=True)
                    raise
                renderers.append(
                    {
                        "channel": channel,
                        "kind": kind,
                        "figure": figure,
                        "source_image": source_image,
                        "video_title": video_title,
                        "time_line": time_line,
                        "writer": writer,
                        "temporary_path": temporary_path,
                        "final_path": final_path,
                    }
                )
                files[channel][kind] = f"videos/{filename}"

        for position, (frame_index, frame_bgr) in enumerate(
            iter_video_frames(video_path, frame_indices, expected_shape)
        ):
            crops = {
                channel: masked_frame_crop(frame_bgr, masks[channel].final_mask)
                for channel in channels
            }
            elapsed = (frame_index - start_frame) / source_fps
            for renderer in renderers:
                renderer["source_image"].set_data(crops[renderer["channel"]])
                renderer["video_title"].set_text(
                    f"{renderer['channel']} · Frame {frame_index} · Elapsed {elapsed:.3f} s"
                )
                renderer["time_line"].set_xdata((position + 0.5, position + 0.5))
                figure = renderer["figure"]
                figure.canvas.draw()
                rgb = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
                renderer["writer"].send(np.ascontiguousarray(rgb).tobytes())

        for renderer in renderers:
            renderer["writer"].close()
            renderer["writer"] = None
        for renderer in renderers:
            renderer["temporary_path"].replace(renderer["final_path"])
        completed = True
    except Exception as exc:
        raise RuntimeError(f"Histogram video encoding failed: {exc}") from exc
    finally:
        for renderer in renderers:
            writer = renderer.get("writer")
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            plt.close(renderer["figure"])
            if not completed:
                renderer["temporary_path"].unlink(missing_ok=True)

    return {
        "enabled": True,
        "selected_types": list(output_kinds),
        "codec": "H.264",
        "pixel_format": "yuv420p",
        "audio": False,
        "resolution": list(VIDEO_FRAME_SIZE),
        "fps": output_fps,
        "frame_count": len(frame_indices),
        "playback": "sampled frames at source FPS divided by frame step",
        "top_panel": "tight analysis-mask crop; pixels outside mask are black",
        "files": files,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path | None:
    video_path = Path(args.video).resolve()
    project_path = Path(args.project).resolve()
    cad_masks_path = Path(args.cad_roi_masks).resolve()
    output_root = Path(args.output_root).resolve()
    raw_video_outputs = list(getattr(args, "video_output", []) or [])
    unknown_video_outputs = [
        name for name in raw_video_outputs if name not in VIDEO_OUTPUT_KINDS
    ]
    if unknown_video_outputs:
        raise ValueError(f"Unknown histogram video outputs: {unknown_video_outputs}")
    video_outputs = tuple(
        name for name in VIDEO_OUTPUT_KINDS if name in raw_video_outputs
    )
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
    histogram_counts: dict[str, list[np.ndarray]] = {channel: [] for channel in channels}
    histogram_percentages: dict[str, list[np.ndarray]] = {
        channel: [] for channel in channels
    }
    for frame_index, frame_bgr in iter_video_frames(video_path, indices, shape):
        for channel in channels:
            calibration_name = "CAL-CH1" if channel.startswith("CH1") else "CAL-CH2"
            result = calculate_phase(
                frame_bgr, masks[channel], averaged_calibrations[calibration_name]
            )
            summary = result.summary()
            if summary["mean_index"] is None:
                raise RuntimeError(f"No valid phase pixels for {channel} at frame {frame_index}")
            counts, percentages = phase_histogram(result.phase[result.valid_mask])
            histogram_counts[channel].append(counts)
            histogram_percentages[channel].append(percentages)
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

    count_matrices = {
        channel: np.stack(histogram_counts[channel], axis=0) for channel in channels
    }
    percent_matrices = {
        channel: np.stack(histogram_percentages[channel], axis=0) for channel in channels
    }

    if args.check:
        print(
            f"Time-series check passed: {len(indices)} frames, "
            f"{indices[0]}..{indices[-1]}, step {args.frame_step}"
        )
        return None

    run_dir = next_run_directory(output_root)
    graph_dir = run_dir / "graphs"
    histogram_dir = run_dir / "histograms"
    graph_dir.mkdir()
    histogram_dir.mkdir()
    save_time_series_csv(run_dir / "time_series.csv", rows)
    save_phase_histogram_csv(
        run_dir / "phase_histogram.csv",
        indices,
        fps,
        args.start_frame,
        channels,
        count_matrices,
        percent_matrices,
    )
    save_time_series_graph(graph_dir / "phase_time_series.png", rows, channels)
    histogram_outputs = save_histogram_heatmaps(
        histogram_dir,
        np.asarray([(index - args.start_frame) / fps for index in indices], dtype=float),
        channels,
        count_matrices,
        percent_matrices,
    )
    histogram_video_outputs: dict[str, Any] = {
        "enabled": False,
        "selected_types": [],
        "files": {},
    }
    if video_outputs:
        histogram_video_outputs = save_histogram_videos(
            run_dir / "videos",
            video_path,
            indices,
            args.start_frame,
            fps,
            args.frame_step,
            channels,
            video_outputs,
            shape,
            masks,
            count_matrices,
            percent_matrices,
        )

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
        "histogram_videos": histogram_video_outputs,
        "histogram": {
            "bin_count": int(PHASE_HISTOGRAM_EDGES.size - 1),
            "bin_width": 0.01,
            "bin_edges": PHASE_HISTOGRAM_EDGES.tolist(),
            "count_unit": "valid pixels per phase bin",
            "percent_normalization": "each frame and channel sums to 100%",
            **histogram_outputs,
        },
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
