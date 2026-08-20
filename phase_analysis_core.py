from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


CHANNEL_SPECS = {
    "CH1-1": {"zone": "CH1-1", "calibration": "CAL-CH1", "flip": False},
    "CH1-2": {"zone": "CH1-2", "calibration": "CAL-CH1", "flip": False},
    "CH2-1": {"zone": "CH2-1", "calibration": "CAL-CH2", "flip": True},
    "CH2-2": {"zone": "CH2-2", "calibration": "CAL-CH2", "flip": True},
}
CHANNEL_ORDER = tuple(CHANNEL_SPECS)
OIL_HUE_CENTER = 0.110
WATER_HUE_CENTER = 0.465
HUE_OFFSET = 0.0
LOW_S_THRESHOLDS = (0.05, 0.10, 0.15)


@dataclass(frozen=True)
class CalibrationResult:
    name: str
    bbox: tuple[int, int, int, int]
    hue_shift: float
    patch_hues: tuple[float, ...]
    patch_saturations: tuple[float, ...]
    matched_reference_hues: tuple[float, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bbox": list(self.bbox),
            "hue_shift": self.hue_shift,
            "patch_hues": list(self.patch_hues),
            "patch_saturations": list(self.patch_saturations),
            "matched_reference_hues": list(self.matched_reference_hues),
        }


@dataclass(frozen=True)
class PhaseResult:
    channel: str
    phase: np.ndarray
    saturation: np.ndarray
    valid_mask: np.ndarray
    column_mean: np.ndarray
    valid_pixel_count: np.ndarray
    cumulative_index: np.ndarray
    low_s_counts: dict[str, int]

    def summary(self) -> dict[str, Any]:
        values = self.phase[self.valid_mask]
        return {
            "channel": self.channel,
            "valid_pixels": int(self.valid_mask.sum()),
            "mean_index": float(np.mean(values)) if values.size else None,
            "std_index": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "final_cumulative_index": finite_last(self.cumulative_index),
            "low_s_counts": self.low_s_counts,
        }


@dataclass(frozen=True)
class WetRefinement:
    drawn_mask: np.ndarray
    proposed_mask: np.ndarray
    change_band: np.ndarray
    protected_foreground: np.ndarray
    added_mask: np.ndarray
    removed_mask: np.ndarray

    def summary(self) -> dict[str, int | float]:
        drawn = int(self.drawn_mask.sum())
        proposed = int(self.proposed_mask.sum())
        added = int(self.added_mask.sum())
        removed = int(self.removed_mask.sum())
        return {
            "drawn_pixels": drawn,
            "proposed_pixels": proposed,
            "added_pixels": added,
            "removed_pixels": removed,
            "changed_pixels": added + removed,
            "changed_percent_of_drawn": 100.0 * (added + removed) / max(1, drawn),
            "protected_foreground_pixels": int(self.protected_foreground.sum()),
            "change_band_pixels": int(self.change_band.sum()),
        }


@dataclass(frozen=True)
class BubbleRefinementItem:
    item_id: str
    drawn_mask: np.ndarray
    proposed_mask: np.ndarray
    used_refinement: bool
    reason: str
    changed_percent: float


@dataclass(frozen=True)
class BubbleRefinement:
    drawn_mask: np.ndarray
    proposed_mask: np.ndarray
    items: tuple[BubbleRefinementItem, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "drawn_pixels": int(self.drawn_mask.sum()),
            "proposed_pixels": int(self.proposed_mask.sum()),
            "refined_items": sum(item.used_refinement for item in self.items),
            "fallback_items": sum(not item.used_refinement for item in self.items),
            "items": [
                {
                    "id": item.item_id,
                    "used_refinement": item.used_refinement,
                    "reason": item.reason,
                    "drawn_pixels": int(item.drawn_mask.sum()),
                    "proposed_pixels": int(item.proposed_mask.sum()),
                    "changed_percent": item.changed_percent,
                }
                for item in self.items
            ],
        }


@dataclass(frozen=True)
class ChannelMasks:
    channel: str
    coarse_roi: np.ndarray
    wet_drawn: np.ndarray
    wet_geometry: np.ndarray
    calibration_excluded: np.ndarray
    island_excluded: np.ndarray
    bubble_excluded: np.ndarray
    artifact_excluded: np.ndarray
    final_mask: np.ndarray

    def summary(self) -> dict[str, int | str]:
        coarse_pixels = int(self.coarse_roi.sum())
        wet_pixels = int(self.wet_geometry.sum())
        calibration_pixels = int(self.calibration_excluded.sum())
        island_pixels = int(self.island_excluded.sum())
        bubble_pixels = int(self.bubble_excluded.sum())
        artifact_pixels = int(self.artifact_excluded.sum())
        final_pixels = int(self.final_mask.sum())
        return {
            "channel": self.channel,
            "coarse_roi_pixels": coarse_pixels,
            "wet_drawn_pixels": int(self.wet_drawn.sum()),
            "wet_geometry_pixels": wet_pixels,
            "outside_wet_pixels": int((self.coarse_roi & ~self.wet_drawn).sum()),
            "calibration_excluded_pixels": calibration_pixels,
            "island_excluded_pixels": island_pixels,
            "bubble_excluded_pixels": bubble_pixels,
            "artifact_excluded_pixels": artifact_pixels,
            "excluded_total_pixels": (
                calibration_pixels + island_pixels + bubble_pixels + artifact_pixels
            ),
            "final_analysis_pixels": final_pixels,
        }

    def validate_partition(self) -> None:
        categories = (
            self.calibration_excluded,
            self.island_excluded,
            self.bubble_excluded,
            self.artifact_excluded,
            self.final_mask,
        )
        accumulated = np.zeros(self.wet_geometry.shape, dtype=np.uint8)
        for category in categories:
            if np.any(accumulated.astype(bool) & category):
                raise AssertionError(f"Overlapping final-mask categories in {self.channel}")
            accumulated += category.astype(np.uint8)
        if not np.array_equal(accumulated.astype(bool), self.wet_geometry):
            raise AssertionError(f"Mask categories do not partition wet geometry in {self.channel}")


def polygon_mask(
    shape: tuple[int, int], points: list[list[float]] | list[tuple[float, float]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        polygon = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def polygon_union_mask(
    shape: tuple[int, int],
    polygons: list[list[list[float]]] | list[list[tuple[float, float]]],
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    valid_polygons = [
        np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
        for points in polygons
        if len(points) >= 3
    ]
    if valid_polygons:
        cv2.fillPoly(result, valid_polygons, 1)
    return result.astype(bool)


def ellipse_kernel(radius: int) -> np.ndarray:
    if radius < 1:
        raise ValueError("Morphology radius must be at least 1")
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.erode(mask.astype(np.uint8), ellipse_kernel(radius)).astype(bool)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), ellipse_kernel(radius)).astype(bool)


def refine_wet_boundary(
    frame_bgr: np.ndarray,
    coarse_roi: np.ndarray,
    drawn_wet: np.ndarray,
    boundary_width: int = 6,
    protection_saturation: float = 0.18,
    ignored_mask: np.ndarray | None = None,
) -> WetRefinement:
    if frame_bgr.shape[:2] != coarse_roi.shape or drawn_wet.shape != coarse_roi.shape:
        raise ValueError("Frame and wet-refinement masks must have equal dimensions")
    if not 2 <= boundary_width <= 20:
        raise ValueError("Boundary width must be between 2 and 20 pixels")
    if not 0.0 <= protection_saturation <= 1.0:
        raise ValueError("Protection saturation must be in [0, 1]")

    coarse = coarse_roi.astype(bool)
    drawn = drawn_wet.astype(bool) & coarse
    if not drawn.any():
        raise ValueError("Cannot refine an empty wet mask")
    ignored = (
        np.zeros_like(coarse)
        if ignored_mask is None
        else ignored_mask.astype(bool) & coarse
    )
    inner_core = erode_mask(drawn, boundary_width)
    outer_limit = dilate_mask(drawn, boundary_width) & coarse
    change_band = (outer_limit & ~inner_core) & ~ignored

    frame_float = frame_bgr.astype(np.float32) / 255.0
    saturation = cv2.cvtColor(frame_float, cv2.COLOR_BGR2HSV)[:, :, 1]
    # Saturation is asymmetric: it may preserve annotated foreground, never reject Low-S fluid.
    protected = drawn & change_band & (saturation >= protection_saturation)

    labels = np.full(coarse.shape, cv2.GC_BGD, dtype=np.uint8)
    labels[coarse & ~drawn] = cv2.GC_PR_BGD
    labels[drawn] = cv2.GC_PR_FGD
    labels[inner_core] = cv2.GC_FGD
    labels[protected] = cv2.GC_FGD
    labels[ignored] = cv2.GC_BGD
    labels[coarse & ~outer_limit] = cv2.GC_BGD

    x0, y0, x1, y1 = channel_bounds(coarse)
    pad = boundary_width + 2
    height, width = coarse.shape
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    local_labels = labels[y0:y1, x0:x1].copy()
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        frame_bgr[y0:y1, x0:x1],
        local_labels,
        None,
        background,
        foreground,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    graphcut = np.zeros_like(coarse)
    graphcut[y0:y1, x0:x1] = (
        (local_labels == cv2.GC_FGD) | (local_labels == cv2.GC_PR_FGD)
    )
    graphcut &= coarse & ~ignored

    # Reject detached fragments: every retained component must touch the hard inner core.
    component_count, components = cv2.connectedComponents(
        graphcut.astype(np.uint8), connectivity=8
    )
    anchored = np.zeros_like(graphcut)
    for component in range(1, component_count):
        candidate = components == component
        if np.any(candidate & inner_core):
            anchored |= candidate

    proposed = drawn.copy()
    proposed[change_band] = anchored[change_band]
    proposed[inner_core | protected] = True
    proposed &= coarse
    added = proposed & ~drawn
    removed = drawn & ~proposed
    return WetRefinement(
        drawn_mask=drawn,
        proposed_mask=proposed,
        change_band=change_band,
        protected_foreground=protected,
        added_mask=added,
        removed_mask=removed,
    )


def wet_refinement_preview_bgr(
    frame_bgr: np.ndarray, refinement: WetRefinement, view: str
) -> np.ndarray:
    result = frame_bgr.copy()
    if view == "refine_drawn":
        result[~refinement.drawn_mask] = (18, 18, 18)
        drawn_boundary = refinement.drawn_mask & ~erode_mask(
            refinement.drawn_mask, 1
        )
        result[drawn_boundary] = (70, 220, 70)
        return result
    if view == "refine_proposed":
        result[~refinement.proposed_mask] = (18, 18, 18)
        proposed_boundary = refinement.proposed_mask & ~erode_mask(
            refinement.proposed_mask, 1
        )
        result[proposed_boundary] = (0, 230, 255)
        return result
    if view == "refine_difference":
        result = np.rint(result.astype(np.float32) * 0.28).astype(np.uint8)
        unchanged = refinement.drawn_mask & refinement.proposed_mask
        result[unchanged] = (50, 115, 50)
        result[refinement.added_mask] = (255, 210, 30)
        result[refinement.removed_mask] = (35, 45, 240)
        protected_boundary = refinement.protected_foreground & ~erode_mask(
            refinement.protected_foreground, 1
        )
        result[protected_boundary] = (0, 230, 255)
        return result
    raise ValueError(f"Unknown wet-refinement preview: {view}")


def refine_bubble_boundaries(
    frame_bgr: np.ndarray,
    wet_mask: np.ndarray,
    exclusions: list[dict[str, Any]],
    boundary_width: int = 3,
    max_changed_percent: float = 22.0,
    smooth_radius: int = 0,
) -> BubbleRefinement:
    if not 2 <= boundary_width <= 10:
        raise ValueError("Bubble boundary width must be between 2 and 10 pixels")
    if not 0 <= smooth_radius <= 8:
        raise ValueError("Bubble smoothing radius must be between 0 and 8 pixels")
    shape = wet_mask.shape
    wet = wet_mask.astype(bool)
    drawn_union = np.zeros(shape, dtype=bool)
    proposed_union = np.zeros(shape, dtype=bool)
    item_results: list[BubbleRefinementItem] = []

    for item in exclusions:
        if str(item.get("kind", "")) != "bubble":
            continue
        item_id = str(item.get("id", "bubble"))
        drawn = polygon_mask(shape, item.get("points", [])) & wet
        drawn_union |= drawn
        if not drawn.any():
            item_results.append(
                BubbleRefinementItem(item_id, drawn, drawn, False, "empty", 0.0)
            )
            continue
        core_radius = max(1, boundary_width - 1)
        core = erode_mask(drawn, core_radius)
        outer_limit = dilate_mask(drawn, boundary_width) & wet
        if not core.any():
            item_results.append(
                BubbleRefinementItem(item_id, drawn, drawn, False, "core_too_small", 0.0)
            )
            proposed_union |= drawn
            continue

        labels = np.full(shape, cv2.GC_BGD, dtype=np.uint8)
        labels[outer_limit] = cv2.GC_PR_BGD
        labels[drawn] = cv2.GC_PR_FGD
        labels[core] = cv2.GC_FGD
        x0, y0, x1, y1 = channel_bounds(outer_limit)
        pad = boundary_width + 2
        height, width = shape
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
        local_labels = labels[y0:y1, x0:x1].copy()
        background = np.zeros((1, 65), dtype=np.float64)
        foreground = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                frame_bgr[y0:y1, x0:x1],
                local_labels,
                None,
                background,
                foreground,
                5,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            item_results.append(
                BubbleRefinementItem(item_id, drawn, drawn, False, "graphcut_failed", 0.0)
            )
            proposed_union |= drawn
            continue

        graphcut = np.zeros(shape, dtype=bool)
        graphcut[y0:y1, x0:x1] = (
            (local_labels == cv2.GC_FGD) | (local_labels == cv2.GC_PR_FGD)
        )
        graphcut &= outer_limit
        count, components = cv2.connectedComponents(graphcut.astype(np.uint8), 8)
        anchored = np.zeros(shape, dtype=bool)
        for component in range(1, count):
            candidate = components == component
            if np.any(candidate & core):
                anchored |= candidate
        anchored[core] = True

        changed = int(np.logical_xor(drawn, anchored).sum())
        changed_percent = 100.0 * changed / max(1, int(drawn.sum()))
        area_ratio = float(anchored.sum()) / max(1, int(drawn.sum()))
        outer_edge = outer_limit & ~erode_mask(outer_limit, 1)
        touches_limit = bool(np.any((anchored & ~drawn) & outer_edge))
        reason = "accepted"
        use_refinement = True
        if changed_percent > max_changed_percent:
            reason, use_refinement = "change_limit", False
        elif not 0.70 <= area_ratio <= 1.30:
            reason, use_refinement = "area_ratio", False
        elif touches_limit:
            reason, use_refinement = "search_limit", False
        proposed = anchored if use_refinement else drawn
        if smooth_radius:
            kernel = ellipse_kernel(smooth_radius)
            rounded = cv2.morphologyEx(
                proposed.astype(np.uint8), cv2.MORPH_OPEN, kernel
            )
            rounded = cv2.morphologyEx(rounded, cv2.MORPH_CLOSE, kernel).astype(bool)
            rounded &= wet
            if rounded.any():
                proposed = rounded
        # A manually marked bubble is a hard exclusion. Refinement and
        # rounding may add pixels to improve the boundary, but must never
        # return any marked wet pixel to the phase result.
        proposed |= drawn
        changed_percent = (
            100.0
            * int(np.logical_xor(drawn, proposed).sum())
            / max(1, int(drawn.sum()))
        )
        proposed_union |= proposed
        item_results.append(
            BubbleRefinementItem(
                item_id,
                drawn,
                proposed,
                use_refinement,
                reason,
                changed_percent,
            )
        )

    return BubbleRefinement(
        drawn_mask=drawn_union,
        proposed_mask=proposed_union,
        items=tuple(item_results),
    )


def bubble_refinement_preview_bgr(
    frame_bgr: np.ndarray, refinement: BubbleRefinement
) -> np.ndarray:
    result = np.rint(frame_bgr.astype(np.float32) * 0.30).astype(np.uint8)
    unchanged = refinement.drawn_mask & refinement.proposed_mask
    added = refinement.proposed_mask & ~refinement.drawn_mask
    removed = refinement.drawn_mask & ~refinement.proposed_mask
    result[unchanged] = (70, 155, 70)
    result[added] = (255, 210, 30)
    result[removed] = (35, 45, 240)
    return result


def capsule_polygon(item: dict[str, Any], template: dict[str, Any]) -> np.ndarray:
    saved_points = np.asarray(item.get("points", []), dtype=np.float64)
    if item.get("shape") != "capsule":
        return saved_points

    center_a = np.asarray(item.get("center_a", []), dtype=np.float64)
    direction_b = np.asarray(item.get("center_b", []), dtype=np.float64)
    if center_a.shape != (2,) or direction_b.shape != (2,):
        return saved_points

    direction = direction_b - center_a
    direction_norm = float(np.linalg.norm(direction))
    radius = float(item.get("radius_px", template.get("radius_px", 0.0)))
    if direction_norm < 1.0 or radius <= 0.0:
        return saved_points

    center_distance = float(template.get("center_distance_px", direction_norm))
    unit = direction / direction_norm
    perpendicular = np.asarray([-unit[1], unit[0]])
    center_b = center_a + unit * center_distance
    points: list[np.ndarray] = []
    for theta in np.linspace(np.pi / 2, -np.pi / 2, 64):
        points.append(
            center_b
            + radius * (unit * np.cos(theta) + perpendicular * np.sin(theta))
        )
    for theta in np.linspace(-np.pi / 2, -3 * np.pi / 2, 64):
        points.append(
            center_a
            + radius * (unit * np.cos(theta) + perpendicular * np.sin(theta))
        )
    return np.asarray(points)


def build_static_exclusion_masks(
    shape: tuple[int, int], roi_project: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.zeros(shape, dtype=np.uint8)
    calibration_polygons = [
        np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
        for points in roi_project.get("calibration_rois", {}).values()
        if len(points) >= 3
    ]
    if calibration_polygons:
        cv2.fillPoly(calibration, calibration_polygons, 1)

    template = roi_project.get("island_template", {})
    scale = max(1, int(template.get("supersample_factor", 8)))
    high_shape = (shape[0] * scale, shape[1] * scale)
    island_high = np.zeros(high_shape, dtype=np.uint8)
    for item in roi_project.get("exclusion_islands", []):
        if not isinstance(item, dict):
            continue
        points = capsule_polygon(item, template)
        if len(points) < 3:
            continue
        polygon = np.rint(points * scale).astype(np.int32)
        cv2.fillPoly(island_high, [polygon], 255)

    islands = np.zeros(shape, dtype=np.uint8)
    if island_high.any():
        coverage = cv2.resize(
            island_high,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        islands[coverage >= 128] = 1
    return calibration.astype(bool), islands.astype(bool)


def manual_exclusion_masks(
    shape: tuple[int, int], exclusions: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    bubbles: list[list[tuple[float, float]]] = []
    artifacts: list[list[tuple[float, float]]] = []
    for item in exclusions:
        kind = str(item.get("kind", ""))
        points = item.get("points", [])
        if kind == "bubble":
            bubbles.append(points)
        elif kind == "artifact":
            artifacts.append(points)
        else:
            raise ValueError(f"Unknown manual exclusion kind: {kind or '<missing>'}")
    return (
        polygon_union_mask(shape, bubbles),
        polygon_union_mask(shape, artifacts),
    )


def compose_channel_masks(
    channel: str,
    coarse_roi: np.ndarray,
    wet_drawn: np.ndarray,
    wet_geometry: np.ndarray,
    calibration_raw: np.ndarray,
    islands_raw: np.ndarray,
    bubbles_raw: np.ndarray,
    artifacts_raw: np.ndarray,
) -> ChannelMasks:
    wet = wet_geometry & coarse_roi
    calibration = wet & calibration_raw
    remaining = wet & ~calibration
    islands = remaining & islands_raw
    remaining &= ~islands
    bubbles = remaining & bubbles_raw
    remaining &= ~bubbles
    artifacts = remaining & artifacts_raw
    final_mask = remaining & ~artifacts
    result = ChannelMasks(
        channel=channel,
        coarse_roi=coarse_roi,
        wet_drawn=wet_drawn,
        wet_geometry=wet,
        calibration_excluded=calibration,
        island_excluded=islands,
        bubble_excluded=bubbles,
        artifact_excluded=artifacts,
        final_mask=final_mask,
    )
    result.validate_partition()
    return result


def build_channel_masks(
    shape: tuple[int, int],
    roi_project: dict[str, Any],
    wet_area_polygons: dict[str, list[list[tuple[float, float]]]],
    manual_exclusions: dict[str, list[dict[str, Any]]],
    channel: str,
    require_wet: bool = False,
    static_masks: tuple[np.ndarray, np.ndarray] | None = None,
    coarse_mask_override: np.ndarray | None = None,
) -> ChannelMasks:
    if channel not in CHANNEL_SPECS:
        raise ValueError(f"Unknown channel: {channel}")
    zone = str(CHANNEL_SPECS[channel]["zone"])
    coarse_roi = polygon_mask(shape, roi_project["rois"][channel]) & polygon_mask(
        shape, roi_project["zones"][zone]
    )
    if coarse_mask_override is not None:
        if coarse_mask_override.shape != shape:
            raise ValueError(
                f"External coarse mask shape for {channel} is "
                f"{coarse_mask_override.shape}, expected {shape}"
            )
        coarse_roi &= coarse_mask_override.astype(bool)
    polygons = wet_area_polygons.get(channel, [])
    if require_wet and not polygons:
        raise ValueError(f"No wet-area polygon saved for {channel}")
    wet_drawn = polygon_union_mask(shape, polygons)
    wet_geometry = coarse_roi & wet_drawn

    calibration_raw, islands_raw = (
        static_masks
        if static_masks is not None
        else build_static_exclusion_masks(shape, roi_project)
    )
    bubbles_raw, artifacts_raw = manual_exclusion_masks(
        shape, manual_exclusions.get(channel, [])
    )

    return compose_channel_masks(
        channel,
        coarse_roi,
        wet_drawn,
        wet_geometry,
        calibration_raw,
        islands_raw,
        bubbles_raw,
        artifacts_raw,
    )


def build_all_channel_masks(
    shape: tuple[int, int],
    roi_project: dict[str, Any],
    wet_area_polygons: dict[str, list[list[tuple[float, float]]]],
    manual_exclusions: dict[str, list[dict[str, Any]]],
    require_wet: bool = False,
    coarse_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, ChannelMasks]:
    static_masks = build_static_exclusion_masks(shape, roi_project)
    return {
        channel: build_channel_masks(
            shape,
            roi_project,
            wet_area_polygons,
            manual_exclusions,
            channel,
            require_wet=require_wet,
            static_masks=static_masks,
            coarse_mask_override=(
                None if coarse_masks is None else coarse_masks[channel]
            ),
        )
        for channel in CHANNEL_ORDER
    }


def mask_qc_bgr(frame_bgr: np.ndarray, masks: ChannelMasks, view: str) -> np.ndarray:
    if view == "original":
        return frame_bgr.copy()

    result = np.zeros_like(frame_bgr)
    if view == "wet":
        result[masks.wet_geometry] = frame_bgr[masks.wet_geometry]
        return result
    if view == "exclusions":
        result[masks.calibration_excluded] = (0, 220, 255)
        result[masks.island_excluded] = (0, 140, 255)
        result[masks.bubble_excluded] = (210, 40, 210)
        result[masks.artifact_excluded] = (40, 40, 240)
        return result
    if view == "final":
        result[masks.final_mask] = frame_bgr[masks.final_mask]
        return result
    raise ValueError(f"Unknown mask QC view: {view}")


def expand_mask_to_cad_height(seed_mask: np.ndarray, cad_region: np.ndarray) -> np.ndarray:
    """Fill complete vertical CAD runs only at x-columns already used by the seed."""
    if seed_mask.shape != cad_region.shape:
        raise ValueError("Seed and CAD masks must have equal dimensions")
    seed = seed_mask.astype(bool)
    cad = cad_region.astype(bool)
    expanded = np.zeros_like(seed)
    for x in np.flatnonzero(seed.any(axis=0)):
        cad_y = np.flatnonzero(cad[:, x])
        seed_y = np.flatnonzero(seed[:, x])
        if cad_y.size == 0 or seed_y.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(cad_y) > 1)
        starts = np.r_[0, breaks + 1]
        stops = np.r_[breaks + 1, cad_y.size]
        selected_any = False
        for start, stop in zip(starts, stops):
            run = cad_y[start:stop]
            if np.any((seed_y >= run[0] - 2) & (seed_y <= run[-1] + 2)):
                expanded[run, x] = True
                selected_any = True
        if not selected_any:
            seed_center = float(np.median(seed_y))
            distances = [
                min(abs(seed_center - cad_y[start]), abs(seed_center - cad_y[stop - 1]))
                for start, stop in zip(starts, stops)
            ]
            nearest = int(np.argmin(distances))
            expanded[cad_y[starts[nearest] : stops[nearest]], x] = True
    return expanded


def circular_difference(measured: float, reference: float) -> float:
    return float((measured - reference + 0.5) % 1.0 - 0.5)


def circular_distance(values: np.ndarray, center: float) -> np.ndarray:
    difference = np.abs(values - center)
    return np.minimum(difference, 1.0 - difference)


def _mean_rgb_hsv(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError(f"Empty calibration patch: {box}")
    mean_rgb = patch.reshape(-1, 3).mean(axis=0)[::-1] / 255.0
    hsv = cv2.cvtColor(
        np.asarray([[mean_rgb]], dtype=np.float32), cv2.COLOR_RGB2HSV
    )[0, 0]
    return float(hsv[0] / 360.0), float(hsv[1])


def calibrate_marked_swatch(
    frame_bgr: np.ndarray,
    roi_project: dict[str, Any],
    name: str,
) -> CalibrationResult:
    points = np.asarray(roi_project.get("calibration_rois", {}).get(name, []), dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError(f"Missing calibration ROI: {name}")
    height, width = frame_bgr.shape[:2]
    left = max(0, int(np.floor(points[:, 0].min())))
    right = min(width, int(np.ceil(points[:, 0].max())) + 1)
    top = max(0, int(np.floor(points[:, 1].min())))
    bottom = min(height, int(np.ceil(points[:, 1].max())) + 1)
    box_width, box_height = right - left, bottom - top
    if box_width < 12 or box_height < 12:
        raise ValueError(f"{name} is too small for 2x2 swatch calibration")

    # Same four interior regions and low-saturation skip rule as the MATLAB source.
    fractions = (
        (0.05, 0.45, 0.10, 0.45),
        (0.55, 0.95, 0.10, 0.45),
        (0.05, 0.45, 0.55, 0.90),
        (0.55, 0.95, 0.55, 0.90),
    )
    reference_hues = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0])
    hues: list[float] = []
    saturations: list[float] = []
    matched: list[float] = []
    shifts: list[float] = []
    for x0f, x1f, y0f, y1f in fractions:
        x0 = left + int(round(x0f * box_width))
        x1 = left + int(round(x1f * box_width))
        y0 = top + int(round(y0f * box_height))
        y1 = top + int(round(y1f * box_height))
        hue, saturation = _mean_rgb_hsv(frame_bgr, (x0, y0, x1, y1))
        if saturation < 0.2:
            continue
        distances = np.minimum(np.abs(reference_hues - hue), 1.0 - np.abs(reference_hues - hue))
        reference = float(reference_hues[int(np.argmin(distances))])
        hues.append(hue)
        saturations.append(saturation)
        matched.append(reference)
        shifts.append(circular_difference(hue, reference))
    if not shifts:
        raise ValueError(f"{name} contains no calibration patch with saturation >= 0.2")
    return CalibrationResult(
        name=name,
        bbox=(left, top, box_width, box_height),
        hue_shift=float(np.mean(shifts)),
        patch_hues=tuple(hues),
        patch_saturations=tuple(saturations),
        matched_reference_hues=tuple(matched),
    )


def calibrate_all(frame_bgr: np.ndarray, roi_project: dict[str, Any]) -> dict[str, CalibrationResult]:
    return {
        name: calibrate_marked_swatch(frame_bgr, roi_project, name)
        for name in ("CAL-CH1", "CAL-CH2")
    }


def column_statistics(
    phase: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.where(valid_mask, phase, 0.0)
    counts = valid_mask.sum(axis=0).astype(np.int64)
    sums = values.sum(axis=0, dtype=np.float64)
    means = np.full(phase.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)

    cumulative = np.full(phase.shape[1], np.nan, dtype=np.float64)
    valid_columns = counts > 0
    if np.any(valid_columns):
        cumulative[valid_columns] = np.cumsum(sums[valid_columns]) / np.cumsum(
            counts[valid_columns]
        )
        indices = np.flatnonzero(np.isfinite(cumulative))
        cumulative[: indices[0]] = cumulative[indices[0]]
        for index in range(indices[0] + 1, cumulative.size):
            if not np.isfinite(cumulative[index]):
                cumulative[index] = cumulative[index - 1]
    return means, counts, cumulative


def finite_last(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else None


def calculate_phase(
    frame_bgr: np.ndarray,
    masks: ChannelMasks,
    calibration: CalibrationResult,
) -> PhaseResult:
    frame_rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hue0 = hsv[:, :, 0] / 360.0
    saturation = hsv[:, :, 1]
    hue = (hue0 - calibration.hue_shift - HUE_OFFSET) % 1.0
    distance_oil = circular_distance(hue, OIL_HUE_CENTER)
    distance_water = circular_distance(hue, WATER_HUE_CENTER)
    fraction = distance_water / (distance_oil + distance_water + np.finfo(np.float32).eps)
    phase = np.full(hue.shape, np.nan, dtype=np.float32)
    phase[masks.final_mask] = 1.0 + fraction[masks.final_mask]
    column_mean, counts, cumulative = column_statistics(phase, masks.final_mask)
    low_s_counts = {
        f"lt_{threshold:.2f}": int((masks.final_mask & (saturation < threshold)).sum())
        for threshold in LOW_S_THRESHOLDS
    }
    return PhaseResult(
        channel=masks.channel,
        phase=phase,
        saturation=saturation,
        valid_mask=masks.final_mask.copy(),
        column_mean=column_mean,
        valid_pixel_count=counts,
        cumulative_index=cumulative,
        low_s_counts=low_s_counts,
    )


def flow_oriented(array: np.ndarray, channel: str) -> np.ndarray:
    if channel not in CHANNEL_SPECS:
        raise ValueError(f"Unknown channel: {channel}")
    return np.fliplr(array) if CHANNEL_SPECS[channel]["flip"] else array.copy()


def channel_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    if not x.size:
        raise ValueError("Cannot crop an empty channel mask")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def crop_to_channel(array: np.ndarray, mask: np.ndarray, channel: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    left, top, right, bottom = channel_bounds(mask)
    crop = array[top:bottom, left:right]
    return flow_oriented(crop, channel), (left, top, right, bottom)
