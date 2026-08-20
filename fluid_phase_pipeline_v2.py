from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_analysis_core import (
    CHANNEL_ORDER,
    HUE_OFFSET,
    LOW_S_THRESHOLDS,
    OIL_HUE_CENTER,
    WATER_HUE_CENTER,
    bubble_refinement_preview_bgr,
    build_all_channel_masks,
    build_static_exclusion_masks,
    calibrate_all,
    calculate_phase,
    channel_bounds,
    column_statistics,
    compose_channel_masks,
    expand_mask_to_cad_height,
    flow_oriented,
    manual_exclusion_masks,
    mask_qc_bgr,
    polygon_mask,
    refine_bubble_boundaries,
    refine_wet_boundary,
    wet_refinement_preview_bgr,
)
from fluid_io import load_analysis_project, load_project, read_video_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fluid phase pipeline V2")
    parser.add_argument("--video", default="double_20260527_144620_000.mp4")
    parser.add_argument(
        "--project", default="projects/double_20260527_144620_000.roi.json"
    )
    parser.add_argument(
        "--analysis-project",
        default="projects/double_20260527_144620_000.phase_v2.json",
    )
    parser.add_argument("--frame", type=int, default=120)
    parser.add_argument("--output-root", default="results/frame_0120_v2")
    parser.add_argument(
        "--cad-roi-masks",
        help="Optional NPZ with CH1-1/CH1-2/CH2-1/CH2-2 CAD coarse masks.",
    )
    parser.add_argument(
        "--cad-placement-config",
        help="Optional CAD placement JSON recorded in summary provenance.",
    )
    parser.add_argument(
        "--cad-outline-lines",
        help="Optional NPZ containing pair_CH1/pair_CH2 complete CAD line masks.",
    )
    parser.add_argument(
        "--cad-vector-lines",
        help="Optional JSON containing transformed CAD vector paths for crisp plotting.",
    )
    parser.add_argument(
        "--equalize-cad-lengths",
        action="store_true",
        help="Trim the longer CH group from figure-right to match the shorter interest-zone length.",
    )
    parser.add_argument(
        "--cad-authoritative",
        action="store_true",
        help="Use CAD channel masks as the wet geometry; apply exclusions afterward.",
    )
    parser.add_argument(
        "--cad-full-height",
        action="store_true",
        help="Keep seed x-extent but fill selected vertical runs to the CAD lines.",
    )
    parser.add_argument(
        "--refine-masks",
        action="store_true",
        help="Refine rough wet/bubble boundaries with conservative quality gates.",
    )
    parser.add_argument("--wet-boundary-width", type=int, default=6)
    parser.add_argument("--bubble-boundary-width", type=int, default=3)
    parser.add_argument("--bubble-max-change", type=float, default=22.0)
    parser.add_argument(
        "--bubble-smooth-radius",
        type=int,
        default=3,
        help="Elliptical morphology radius used to round final bubble masks.",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_coarse_masks(
    path: Path, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as payload:
        missing = [channel for channel in CHANNEL_ORDER if channel not in payload]
        if missing:
            raise ValueError(f"CAD ROI mask file is missing: {', '.join(missing)}")
        for channel in CHANNEL_ORDER:
            mask = np.asarray(payload[channel]).astype(bool)
            if mask.shape != shape:
                raise ValueError(
                    f"CAD ROI mask shape for {channel} is {mask.shape}, expected {shape}"
                )
            if not mask.any():
                raise ValueError(f"CAD ROI mask is empty for {channel}")
            masks[channel] = mask
    return masks


def load_external_cad_lines(
    path: Path, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    lines: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as payload:
        for group in ("CH1", "CH2"):
            key = f"pair_{group}"
            if key not in payload:
                raise ValueError(f"CAD outline file is missing: {key}")
            line_mask = np.asarray(payload[key]).astype(bool)
            if line_mask.shape != shape:
                raise ValueError(
                    f"CAD outline shape for {group} is {line_mask.shape}, expected {shape}"
                )
            lines[group] = line_mask
    return lines


def load_external_cad_vector_lines(
    path: Path,
) -> dict[str, list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported CAD vector-line schema: {path}")
    groups = payload.get("groups", {})
    result: dict[str, list[np.ndarray]] = {}
    for group in ("CH1", "CH2"):
        if group not in groups:
            raise ValueError(f"CAD vector-line file is missing {group}: {path}")
        paths: list[np.ndarray] = []
        for points in groups[group]:
            array = np.asarray(points, dtype=np.float64)
            if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] == 2:
                paths.append(array)
        result[group] = paths
    return result


def _clip_segment_to_rect(
    start: np.ndarray,
    end: np.ndarray,
    rectangle: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang-Barsky clipping in floating-point image coordinates."""
    xmin, ymin, xmax, ymax = rectangle
    delta = end - start
    p = (-delta[0], delta[0], -delta[1], delta[1])
    q = (start[0] - xmin, xmax - start[0], start[1] - ymin, ymax - start[1])
    lower, upper = 0.0, 1.0
    for direction, distance in zip(p, q):
        if abs(float(direction)) < 1e-12:
            if distance < 0:
                return None
            continue
        ratio = float(distance / direction)
        if direction < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return start + lower * delta, start + upper * delta


def vector_paths_for_flow_crop(
    paths: list[np.ndarray],
    bounds: tuple[int, int, int, int],
    flip_horizontal: bool,
    clip_rectangle: tuple[float, float, float, float] | None = None,
) -> list[np.ndarray]:
    left, top, right, bottom = bounds
    rectangle = (float(left), float(top), float(right - 1), float(bottom - 1))
    if clip_rectangle is not None:
        rectangle = (
            max(rectangle[0], clip_rectangle[0]),
            max(rectangle[1], clip_rectangle[1]),
            min(rectangle[2], clip_rectangle[2]),
            min(rectangle[3], clip_rectangle[3]),
        )
    width = right - left
    clipped_paths: list[np.ndarray] = []
    for path in paths:
        for start, end in zip(path[:-1], path[1:]):
            clipped = _clip_segment_to_rect(start, end, rectangle)
            if clipped is None:
                continue
            segment = np.asarray(clipped, dtype=np.float64)
            segment[:, 0] -= left
            segment[:, 1] -= top
            if flip_horizontal:
                segment[:, 0] = (width - 1) - segment[:, 0]
                segment = segment[::-1]
            clipped_paths.append(segment)
    return clipped_paths


def channel_vector_rectangle(
    roi_project: dict[str, Any],
    channel: str,
    shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    group = channel.split("-", 1)[0]
    channels = (f"{group}-1", f"{group}-2")
    zone = np.asarray(roi_project["zones"][f"ZONE-{group}"], dtype=float)
    first_center = float(
        np.asarray(roi_project["rois"][channels[0]], dtype=float)[:, 1].mean()
    )
    second_center = float(
        np.asarray(roi_project["rois"][channels[1]], dtype=float)[:, 1].mean()
    )
    separator = (first_center + second_center) / 2.0
    xmin, xmax = float(zone[:, 0].min()), float(zone[:, 0].max())
    if (channel == channels[0]) == (first_center <= second_center):
        return xmin, 0.0, xmax, separator
    return xmin, separator, xmax, float(shape[0] - 1)


def equalize_cad_interest_zone_lengths(
    roi_project: dict[str, Any],
) -> dict[str, Any]:
    measurements: dict[str, tuple[float, float, int]] = {}
    for group in ("CH1", "CH2"):
        zone = np.asarray(roi_project["zones"][f"ZONE-{group}"], dtype=float)
        xmin, xmax = float(zone[:, 0].min()), float(zone[:, 0].max())
        measurements[group] = (xmin, xmax, int(round(xmax - xmin)) + 1)
    target = min(item[2] for item in measurements.values())
    report: dict[str, Any] = {"target_width_px": target, "groups": {}}
    for group, (xmin, xmax, width) in measurements.items():
        trim = max(0, width - target)
        # CH2 is horizontally flipped in the publication view, so its
        # figure-right side is frame-left.  CH1 figure-right is frame-right.
        if group == "CH2":
            effective_min, effective_max = xmin + trim, xmax
        else:
            effective_min, effective_max = xmin, xmax - trim
        points = roi_project["zones"][f"ZONE-{group}"]
        roi_project["zones"][f"ZONE-{group}"] = [
            [
                max(effective_min, min(effective_max, float(x))),
                float(y),
            ]
            for x, y in points
        ]
        report["groups"][group] = {
            "original_width_px": width,
            "trimmed_from_figure_right_px": trim,
            "effective_x_range": [effective_min, effective_max],
        }
    return report


def vector_bounds_in_rectangle(
    paths: list[np.ndarray],
    rectangle: tuple[float, float, float, float],
) -> tuple[int, int, int, int] | None:
    points: list[np.ndarray] = []
    for path in paths:
        for start, end in zip(path[:-1], path[1:]):
            clipped = _clip_segment_to_rect(start, end, rectangle)
            if clipped is not None:
                points.extend(clipped)
    if not points:
        return None
    array = np.asarray(points, dtype=float)
    left = max(0, int(np.floor(array[:, 0].min())))
    top = max(0, int(np.floor(array[:, 1].min())))
    right = int(np.ceil(array[:, 0].max())) + 1
    bottom = int(np.ceil(array[:, 1].max())) + 1
    return left, top, right, bottom


def channel_cad_lines_in_interest_zones(
    pair_lines: dict[str, np.ndarray],
    roi_project: dict[str, Any],
    shape: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Clip CAD lines by interest zone, then separate each stacked channel.

    ROI geometry is used only to locate the separator between the two stacked
    channels.  It never clips the CAD line itself.
    """
    rows = np.indices(shape)[0]
    clipped_pairs: dict[str, np.ndarray] = {}
    channel_lines: dict[str, np.ndarray] = {}
    for group, channels in {
        "CH1": ("CH1-1", "CH1-2"),
        "CH2": ("CH2-1", "CH2-2"),
    }.items():
        zone_name = f"ZONE-{group}"
        clipped = pair_lines[group] & polygon_mask(
            shape, roi_project["zones"][zone_name]
        )
        clipped_pairs[group] = clipped
        first_center_y = float(
            np.asarray(roi_project["rois"][channels[0]], dtype=float)[:, 1].mean()
        )
        second_center_y = float(
            np.asarray(roi_project["rois"][channels[1]], dtype=float)[:, 1].mean()
        )
        separator_y = (first_center_y + second_center_y) / 2.0
        if first_center_y <= second_center_y:
            channel_lines[channels[0]] = clipped & (rows <= separator_y)
            channel_lines[channels[1]] = clipped & (rows > separator_y)
        else:
            channel_lines[channels[0]] = clipped & (rows > separator_y)
            channel_lines[channels[1]] = clipped & (rows <= separator_y)
    return clipped_pairs, channel_lines


def next_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    number = 1
    while (output_root / f"run_{number:03d}").exists():
        number += 1
    path = output_root / f"run_{number:03d}"
    path.mkdir()
    return path


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def save_summary_csv(path: Path, summaries: dict[str, Any]) -> None:
    fields = (
        "channel",
        "valid_pixels",
        "mean_index",
        "std_index",
        "final_cumulative_index",
        "wet_pixels",
        "excluded_pixels",
        "low_s_lt_005",
        "low_s_lt_010",
        "low_s_lt_015",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for channel in CHANNEL_ORDER:
            phase = summaries[channel]["phase"]
            mask = summaries[channel]["mask"]
            low_s = phase["low_s_counts"]
            writer.writerow(
                {
                    "channel": channel,
                    "valid_pixels": phase["valid_pixels"],
                    "mean_index": phase["mean_index"],
                    "std_index": phase["std_index"],
                    "final_cumulative_index": phase["final_cumulative_index"],
                    "wet_pixels": mask["wet_geometry_pixels"],
                    "excluded_pixels": mask["excluded_total_pixels"],
                    "low_s_lt_005": low_s["lt_0.05"],
                    "low_s_lt_010": low_s["lt_0.10"],
                    "low_s_lt_015": low_s["lt_0.15"],
                }
            )


def save_mask(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Cannot write {path}")


def save_phase_color(
    path: Path,
    phase: np.ndarray,
    valid: np.ndarray,
    outline: np.ndarray | None = None,
) -> None:
    normalized = np.clip(phase - 1.0, 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(np.nan_to_num(normalized, nan=0.0))
    rgb = np.rint(rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    if outline is not None:
        rgb[outline.astype(bool)] = 255
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Cannot write {path}")


def save_channel_graph(path: Path, channel: str, means: np.ndarray, counts: np.ndarray, cumulative: np.ndarray) -> None:
    x = np.arange(means.size)
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot(x, means, color="#9a9a9a", linewidth=1, label="Column mean")
    top.plot(x, cumulative, color="#c62828", linewidth=1.8, label="Cumulative")
    top.axhline(1.5, color="#222222", linestyle="--", linewidth=0.9)
    top.set_ylim(1.0, 2.0)
    top.set_ylabel("Phase index")
    top.set_title(channel)
    top.grid(alpha=0.2)
    top.legend(frameon=False)
    bottom.fill_between(x, counts, color="#1976a3", alpha=0.85)
    bottom.set_ylabel("Valid px")
    bottom.set_xlabel("Flow-oriented axial position (px)")
    bottom.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_combined_graph(path: Path, flow_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for channel, (_, _, cumulative) in flow_results.items():
        axis.plot(np.linspace(0.0, 1.0, cumulative.size), cumulative, linewidth=1.8, label=channel)
    axis.axhline(1.5, color="#222222", linestyle="--", linewidth=0.9)
    axis.set(xlabel="Normalized flow position", ylabel="Cumulative phase index", ylim=(1.0, 2.0))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_publication_image(
    path_png: Path,
    path_pdf: Path,
    flow_crops: dict[str, tuple[np.ndarray, np.ndarray]],
    flow_outlines: dict[str, np.ndarray] | None = None,
    flow_vector_lines: dict[str, list[np.ndarray]] | None = None,
) -> None:
    max_width = max(phase.shape[1] for phase, _ in flow_crops.values())
    gap = 8
    heights = [phase.shape[0] for phase, _ in flow_crops.values()]
    footer = 28
    canvas = np.full((sum(heights) + gap * 3 + footer, max_width), np.nan, dtype=np.float32)
    outline_canvas = np.zeros(canvas.shape, dtype=bool)
    y = 0
    centers: list[float] = []
    vector_canvas: list[np.ndarray] = []
    for channel, (phase, valid) in flow_crops.items():
        height, width = phase.shape
        canvas[y : y + height, :width][valid] = phase[valid]
        if flow_outlines is not None:
            outline_canvas[y : y + height, :width] = flow_outlines[channel]
        if flow_vector_lines is not None:
            for path in flow_vector_lines[channel]:
                shifted = path.copy()
                shifted[:, 1] += y
                vector_canvas.append(shifted)
        centers.append(y + height / 2)
        y += height + gap

    figure_height = max(5.2, min(12.0, 11.0 * canvas.shape[0] / max_width))
    figure, axis = plt.subplots(figsize=(11, figure_height))
    image = axis.imshow(
        canvas,
        cmap="turbo",
        vmin=1.0,
        vmax=2.0,
        interpolation="bilinear",
        resample=True,
    )
    if flow_outlines is not None:
        outline_rgba = np.zeros((*outline_canvas.shape, 4), dtype=np.uint8)
        outline_rgba[outline_canvas] = (0, 0, 0, 255)
        axis.imshow(outline_rgba, interpolation="nearest")
    if flow_vector_lines is not None:
        for path in vector_canvas:
            axis.plot(
                path[:, 0],
                path[:, 1],
                color="black",
                linewidth=1.7,
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
    axis.set_yticks(centers, CHANNEL_ORDER)
    axis.set_xticks([])
    axis.tick_params(axis="y", length=0, pad=8)
    for spine in axis.spines.values():
        spine.set_visible(False)
    scale_length = min(100, max(20, max_width // 5))
    scale_y = canvas.shape[0] - 10
    axis.plot([5, 5 + scale_length], [scale_y, scale_y], color="black", linewidth=2.5)
    axis.text(5 + scale_length / 2, scale_y - 4, f"{scale_length} px", color="black", ha="center", va="bottom", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Phase index (1 = water, 2 = oil)")
    figure.tight_layout()
    figure.savefig(path_png, dpi=450, transparent=False, bbox_inches="tight")
    figure.savefig(path_pdf, dpi=450, transparent=False, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path | None:
    video_path = Path(args.video).resolve()
    roi_path = Path(args.project).resolve()
    analysis_path = Path(args.analysis_project).resolve()
    output_root = Path(args.output_root).resolve()
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Analysis project not found: {analysis_path}. Draw and explicitly save all wet areas first."
        )

    frame = read_video_frame(video_path, args.frame)
    roi_project = load_project(roi_path, frame.bgr.shape[:2])
    cad_length_equalization = (
        equalize_cad_interest_zone_lengths(roi_project)
        if args.equalize_cad_lengths
        else None
    )
    roi_hash = file_sha256(roi_path)
    cad_mask_path = Path(args.cad_roi_masks).resolve() if args.cad_roi_masks else None
    cad_masks = (
        load_external_coarse_masks(cad_mask_path, frame.bgr.shape[:2])
        if cad_mask_path is not None
        else None
    )
    if args.cad_authoritative and cad_masks is None:
        raise ValueError("--cad-authoritative requires --cad-roi-masks")
    if args.cad_full_height and cad_masks is None:
        raise ValueError("--cad-full-height requires --cad-roi-masks")
    if args.cad_authoritative and args.cad_full_height:
        raise ValueError("Choose either --cad-authoritative or --cad-full-height")
    cad_line_path = Path(args.cad_outline_lines).resolve() if args.cad_outline_lines else None
    cad_lines = (
        load_external_cad_lines(cad_line_path, frame.bgr.shape[:2])
        if cad_line_path is not None
        else None
    )
    cad_channel_lines: dict[str, np.ndarray] | None = None
    if cad_lines is not None:
        cad_lines, cad_channel_lines = channel_cad_lines_in_interest_zones(
            cad_lines, roi_project, frame.bgr.shape[:2]
        )
    cad_vector_path = (
        Path(args.cad_vector_lines).resolve() if args.cad_vector_lines else None
    )
    cad_vector_lines = (
        load_external_cad_vector_lines(cad_vector_path)
        if cad_vector_path is not None
        else None
    )
    analysis = load_analysis_project(
        analysis_path, roi_path, roi_hash, frame.bgr.shape[:2], args.frame
    )
    if args.cad_authoritative:
        calibration_mask, island_mask = build_static_exclusion_masks(
            frame.bgr.shape[:2], roi_project
        )
        # CAD internal loops are authoritative in this mode.  Do not subtract
        # the legacy hand-placed island ROIs a second time.
        island_mask = np.zeros_like(island_mask, dtype=bool)
        drawn_masks = {}
        for channel in CHANNEL_ORDER:
            bubbles, artifacts = manual_exclusion_masks(
                frame.bgr.shape[:2], analysis.manual_exclusions[channel]
            )
            zone_name = "ZONE-CH1" if channel.startswith("CH1") else "ZONE-CH2"
            cad_geometry = cad_masks[channel] & polygon_mask(
                frame.bgr.shape[:2], roi_project["zones"][zone_name]
            )
            drawn_masks[channel] = compose_channel_masks(
                channel,
                cad_geometry,
                cad_geometry,
                cad_geometry,
                calibration_mask,
                island_mask,
                bubbles,
                artifacts,
            )
    else:
        drawn_masks = build_all_channel_masks(
            frame.bgr.shape[:2],
            roi_project,
            analysis.wet_area_polygons,
            analysis.manual_exclusions,
            require_wet=True,
            coarse_masks=cad_masks,
        )
        if args.cad_full_height:
            calibration_mask, island_mask = build_static_exclusion_masks(
                frame.bgr.shape[:2], roi_project
            )
            expanded_masks = {}
            for channel in CHANNEL_ORDER:
                base = drawn_masks[channel]
                bubbles, artifacts = manual_exclusion_masks(
                    frame.bgr.shape[:2], analysis.manual_exclusions[channel]
                )
                expanded_wet = expand_mask_to_cad_height(
                    base.wet_geometry, cad_masks[channel]
                )
                expanded_masks[channel] = compose_channel_masks(
                    channel,
                    cad_masks[channel],
                    base.wet_drawn,
                    expanded_wet,
                    calibration_mask,
                    island_mask,
                    bubbles,
                    artifacts,
                )
            drawn_masks = expanded_masks
    masks = drawn_masks
    refinement_reports: dict[str, Any] = {}
    wet_refinements: dict[str, Any] = {}
    bubble_refinements: dict[str, Any] = {}
    if args.refine_masks:
        calibration_mask, island_mask = build_static_exclusion_masks(
            frame.bgr.shape[:2], roi_project
        )
        if args.cad_authoritative:
            island_mask = np.zeros_like(island_mask, dtype=bool)
        refined_masks = {}
        for channel in CHANNEL_ORDER:
            base = drawn_masks[channel]
            wet_refinement = None
            refined_wet = base.wet_geometry
            if not (args.cad_authoritative or args.cad_full_height):
                wet_refinement = refine_wet_boundary(
                    frame.bgr,
                    base.coarse_roi,
                    base.wet_geometry,
                    boundary_width=args.wet_boundary_width,
                    ignored_mask=calibration_mask | island_mask,
                )
                refined_wet = wet_refinement.proposed_mask
            bubble_refinement = refine_bubble_boundaries(
                frame.bgr,
                refined_wet,
                analysis.manual_exclusions[channel],
                boundary_width=args.bubble_boundary_width,
                max_changed_percent=args.bubble_max_change,
                smooth_radius=args.bubble_smooth_radius,
            )
            _, artifact_mask = manual_exclusion_masks(
                frame.bgr.shape[:2], analysis.manual_exclusions[channel]
            )
            refined_masks[channel] = compose_channel_masks(
                channel,
                base.coarse_roi,
                base.wet_drawn,
                refined_wet,
                calibration_mask,
                island_mask,
                bubble_refinement.proposed_mask,
                artifact_mask,
            )
            if wet_refinement is not None:
                wet_refinements[channel] = wet_refinement
            bubble_refinements[channel] = bubble_refinement
            refinement_reports[channel] = {
                "wet": (
                    wet_refinement.summary()
                    if wet_refinement is not None
                    else {
                        "mode": (
                            "CAD full-height; wet refinement skipped"
                            if args.cad_full_height
                            else "CAD authoritative; wet refinement skipped"
                        )
                    }
                ),
                "bubble": bubble_refinement.summary(),
            }
        masks = refined_masks
    calibrations = calibrate_all(frame.bgr, roi_project)
    phase_results = {
        channel: calculate_phase(
            frame.bgr,
            masks[channel],
            calibrations["CAL-CH1" if channel.startswith("CH1") else "CAL-CH2"],
        )
        for channel in CHANNEL_ORDER
    }
    if args.check:
        for channel in CHANNEL_ORDER:
            print(channel, masks[channel].summary(), phase_results[channel].summary())
            if args.refine_masks:
                print("  refinement", refinement_reports[channel])
        return None

    run_dir = next_run_directory(output_root)
    qc_dir = run_dir / "qc"
    array_dir = run_dir / "arrays"
    graph_dir = run_dir / "graphs"
    for directory in (qc_dir, array_dir, graph_dir):
        directory.mkdir()

    if cad_lines is not None:
        complete_outline = cad_lines["CH1"] | cad_lines["CH2"]
        np.save(array_dir / "cad_outline_complete.npy", complete_outline)
        save_mask(array_dir / "cad_outline_complete.png", complete_outline)
        outline_preview = frame.bgr.copy()
        outline_preview[complete_outline] = (255, 255, 255)
        cv2.imwrite(str(qc_dir / "cad_outline_overlay.png"), outline_preview)

    flow_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    flow_crops: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    flow_outlines: dict[str, np.ndarray] = {}
    flow_vector_lines: dict[str, list[np.ndarray]] = {}
    summaries: dict[str, Any] = {}
    reassembled_sum = np.zeros(frame.bgr.shape[:2], dtype=np.float64)
    reassembled_count = np.zeros(frame.bgr.shape[:2], dtype=np.uint8)

    for channel in CHANNEL_ORDER:
        channel_masks = masks[channel]
        result = phase_results[channel]
        np.save(array_dir / f"phase_raw_{channel}.npy", result.phase)
        np.save(array_dir / f"phase_zero_{channel}.npy", np.nan_to_num(result.phase, nan=0.0))
        np.save(array_dir / f"valid_mask_{channel}.npy", result.valid_mask)
        np.save(array_dir / f"wet_mask_drawn_{channel}.npy", drawn_masks[channel].wet_geometry)
        np.save(array_dir / f"wet_mask_used_{channel}.npy", channel_masks.wet_geometry)
        np.save(array_dir / f"bubble_mask_drawn_{channel}.npy", drawn_masks[channel].bubble_excluded)
        np.save(array_dir / f"bubble_mask_used_{channel}.npy", channel_masks.bubble_excluded)
        save_mask(array_dir / f"valid_mask_{channel}.png", result.valid_mask)
        if cad_masks is not None:
            np.save(array_dir / f"cad_roi_mask_{channel}.npy", cad_masks[channel])
            save_mask(array_dir / f"cad_roi_mask_{channel}.png", cad_masks[channel])
            cad_preview = np.zeros_like(frame.bgr)
            cad_preview[cad_masks[channel]] = frame.bgr[cad_masks[channel]]
            cv2.imwrite(str(qc_dir / f"cad_roi_{channel}.png"), cad_preview)
        channel_outline = (
            cad_channel_lines[channel] if cad_channel_lines is not None else None
        )
        save_phase_color(
            qc_dir / f"phase_{channel}.png",
            result.phase,
            result.valid_mask,
            channel_outline,
        )
        for view in ("wet", "exclusions", "final"):
            cv2.imwrite(str(qc_dir / f"mask_{view}_{channel}.png"), mask_qc_bgr(frame.bgr, channel_masks, view))
        if args.refine_masks and channel in wet_refinements:
            cv2.imwrite(
                str(qc_dir / f"wet_refinement_difference_{channel}.png"),
                wet_refinement_preview_bgr(
                    frame.bgr, wet_refinements[channel], "refine_difference"
                ),
            )
            cv2.imwrite(
                str(qc_dir / f"bubble_refinement_difference_{channel}.png"),
                bubble_refinement_preview_bgr(
                    frame.bgr, bubble_refinements[channel]
                ),
            )

        valid_values = np.nan_to_num(result.phase, nan=0.0)
        reassembled_sum[result.valid_mask] += valid_values[result.valid_mask]
        reassembled_count[result.valid_mask] += 1

        left, top, right, bottom = channel_bounds(result.valid_mask)
        if channel_outline is not None:
            outline_rows = np.flatnonzero(channel_outline.any(axis=1))
            outline_columns = np.flatnonzero(channel_outline.any(axis=0))
            if outline_rows.size and outline_columns.size:
                left = min(left, int(outline_columns.min()))
                right = max(right, int(outline_columns.max()) + 1)
                top = min(top, int(outline_rows.min()))
                bottom = max(bottom, int(outline_rows.max()) + 1)
        vector_rectangle = None
        if cad_vector_lines is not None:
            vector_rectangle = channel_vector_rectangle(
                roi_project, channel, frame.bgr.shape[:2]
            )
            vector_bounds = vector_bounds_in_rectangle(
                cad_vector_lines[channel.split("-", 1)[0]], vector_rectangle
            )
            if vector_bounds is not None:
                vector_left, vector_top, vector_right, vector_bottom = vector_bounds
                left = min(left, vector_left)
                top = min(top, vector_top)
                right = max(right, vector_right)
                bottom = max(bottom, vector_bottom)
        flow_phase = flow_oriented(result.phase[top:bottom, left:right], channel)
        flow_valid = flow_oriented(result.valid_mask[top:bottom, left:right], channel)
        means, counts, cumulative = column_statistics(flow_phase, flow_valid)
        flow_results[channel] = (means, counts, cumulative)
        flow_crops[channel] = (flow_phase, flow_valid)
        if channel_outline is not None:
            flow_outlines[channel] = flow_oriented(
                channel_outline[top:bottom, left:right], channel
            )
        if cad_vector_lines is not None:
            flow_vector_lines[channel] = vector_paths_for_flow_crop(
                cad_vector_lines[channel.split("-", 1)[0]],
                (left, top, right, bottom),
                flip_horizontal=channel.startswith("CH2"),
                clip_rectangle=vector_rectangle,
            )
        save_channel_graph(graph_dir / f"phase_profile_{channel}.png", channel, means, counts, cumulative)

        sensitivity: dict[str, Any] = {}
        primary_final = result.summary()["final_cumulative_index"]
        flow_saturation = flow_oriented(result.saturation[top:bottom, left:right], channel)
        for threshold in LOW_S_THRESHOLDS:
            sensitivity_valid = flow_valid & (flow_saturation >= threshold)
            _, sensitivity_counts, sensitivity_cumulative = column_statistics(flow_phase, sensitivity_valid)
            finite = sensitivity_cumulative[np.isfinite(sensitivity_cumulative)]
            final = float(finite[-1]) if finite.size else None
            difference = abs(final - primary_final) if final is not None and primary_final is not None else None
            sensitivity[f"S >= {threshold:.2f}"] = {
                "valid_pixels": int(sensitivity_counts.sum()),
                "final_cumulative_index": final,
                "absolute_difference_from_primary": difference,
                "warning_gt_0.02": bool(difference is not None and difference > 0.02),
            }
        summaries[channel] = {
            "mask": channel_masks.summary(),
            "phase": result.summary(),
            "flow_crop_original_bbox": [left, top, right, bottom],
            "flow_flipped": channel.startswith("CH2"),
            "low_s_sensitivity": sensitivity,
            "mask_refinement": refinement_reports.get(channel),
        }

    reassembled = np.full(frame.bgr.shape[:2], np.nan, dtype=np.float32)
    overlap = reassembled_count > 0
    reassembled[overlap] = (reassembled_sum[overlap] / reassembled_count[overlap]).astype(np.float32)
    np.save(array_dir / "phase_reassembled_original_coordinates.npy", reassembled)
    np.save(array_dir / "phase_reassembled_zero.npy", np.nan_to_num(reassembled, nan=0.0))
    np.save(array_dir / "valid_mask_reassembled.npy", overlap)
    save_mask(array_dir / "valid_mask_reassembled.png", overlap)
    save_phase_color(qc_dir / "phase_reassembled_original_coordinates.png", reassembled, overlap)
    save_combined_graph(graph_dir / "combined_cumulative.png", flow_results)
    save_publication_image(
        graph_dir / "phase_publication_aligned_left.png",
        graph_dir / "phase_publication_aligned_left.pdf",
        flow_crops,
        flow_outlines if cad_lines is not None and cad_vector_lines is None else None,
        flow_vector_lines if cad_vector_lines is not None else None,
    )

    summary = {
        "pipeline": "fluid_phase_pipeline_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame": args.frame,
        "video": {"path": str(video_path), "sha256": file_sha256(video_path)},
        "roi_project": {"path": str(roi_path), "sha256": roi_hash},
        "analysis_project": {"path": str(analysis_path), "sha256": file_sha256(analysis_path)},
        "cad_roi": (
            {
                "masks_path": str(cad_mask_path),
                "masks_sha256": file_sha256(cad_mask_path),
                "authoritative_geometry": bool(args.cad_authoritative),
                "outline_lines": (
                    {
                        "path": str(cad_line_path),
                        "sha256": file_sha256(cad_line_path),
                    }
                    if cad_line_path is not None
                    else None
                ),
                "vector_lines": (
                    {
                        "path": str(cad_vector_path),
                        "sha256": file_sha256(cad_vector_path),
                    }
                    if cad_vector_path is not None
                    else None
                ),
                "placement_config": (
                    {
                        "path": str(Path(args.cad_placement_config).resolve()),
                        "sha256": file_sha256(Path(args.cad_placement_config).resolve()),
                    }
                    if args.cad_placement_config
                    else None
                ),
                "rule": (
                    "CAD interior and CAD holes AND channel ROI AND interest zone, then calibration/bubble/artifact exclusions; legacy island ROIs disabled"
                    if args.cad_authoritative
                    else "legacy ROI AND zone AND CAD pair mask, then wet/exclusions"
                ),
                "outline_rule": (
                    "channel CAD line AND interest zone; ROI does not clip the outline"
                    if cad_line_path is not None
                    else None
                ),
                "length_equalization": cad_length_equalization,
            }
            if cad_mask_path is not None
            else None
        ),
        "parameters": {
            "oil_hue_center": OIL_HUE_CENTER,
            "water_hue_center": WATER_HUE_CENTER,
            "hue_offset": HUE_OFFSET,
            "smoothing": None,
            "air_detector": None,
            "low_s_primary_policy": "included",
            "low_s_qc_thresholds": list(LOW_S_THRESHOLDS),
            "mask_refinement": {
                "enabled": bool(args.refine_masks),
                "wet_boundary_width_px": args.wet_boundary_width,
                "bubble_boundary_width_px": args.bubble_boundary_width,
                "bubble_max_changed_percent": args.bubble_max_change,
                "bubble_smooth_radius_px": args.bubble_smooth_radius,
                "bubble_quality_gate": "per polygon; fallback on change, area ratio, or search-limit failure",
            },
        },
        "calibrations": {name: value.summary() for name, value in calibrations.items()},
        "channels": summaries,
        "reassembled_overlap_pixels": int((reassembled_count > 1).sum()),
    }
    save_json(run_dir / "summary.json", summary)
    save_summary_csv(run_dir / "summary.csv", summaries)
    print(f"V2 result: {run_dir}")
    return run_dir


def main() -> None:
    try:
        run(parse_args())
    except Exception as error:
        print(
            f"V2 pipeline stopped ({type(error).__name__}): {error}",
            file=sys.stdout,
        )
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
    bubble_refinement_preview_bgr,
