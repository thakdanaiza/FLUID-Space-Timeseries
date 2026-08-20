from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


CHANNEL_GROUPS = {
    "CH1": ("CH1-1", "CH1-2"),
    "CH2": ("CH2-1", "CH2-2"),
}


def polygon_mask(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    polygon = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def enclosed_regions(line_mask: np.ndarray) -> list[np.ndarray]:
    barrier = cv2.morphologyEx(
        line_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    count, labels = cv2.connectedComponents((~barrier).astype(np.uint8), 8)
    border_labels = set(
        np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist()
    )
    return [
        labels == label
        for label in range(1, count)
        if label not in border_labels and int((labels == label).sum()) >= 100
    ]


def build_cad_masks(
    lines_path: Path,
    roi_project: dict[str, Any],
    output_path: Path,
) -> dict[str, np.ndarray]:
    shape = (
        int(roi_project["source"]["frame_height"]),
        int(roi_project["source"]["frame_width"]),
    )
    masks: dict[str, np.ndarray] = {}
    selected_channels = set(
        roi_project.get("channels", [channel for channels in CHANNEL_GROUPS.values() for channel in channels])
    )
    with np.load(lines_path, allow_pickle=False) as payload:
        for group, channels in CHANNEL_GROUPS.items():
            requested_channels = [channel for channel in channels if channel in selected_channels]
            if not requested_channels:
                continue
            line_mask = np.asarray(payload[f"pair_{group}"]).astype(bool)
            if line_mask.shape != shape:
                raise ValueError(f"CAD line dimensions differ for {group}")
            regions = enclosed_regions(line_mask)
            if not regions:
                raise ValueError(f"No enclosed CAD region found for {group}")
            seeds = [
                polygon_mask(shape, roi_project["rois"][channel])
                & polygon_mask(shape, roi_project["zones"][channel])
                for channel in requested_channels
            ]
            scores = np.asarray([[int((region & seed).sum()) for region in regions] for seed in seeds])
            flow_index = int(np.argmax(scores[0]))
            if int(scores[0, flow_index]) <= 0:
                raise ValueError(f"ROI for {group} does not overlap the enclosed CAD area")
            enclosed = regions[flow_index]
            inner_stroke = line_mask & (
                cv2.dilate(
                    enclosed.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                )
                > 0
            )
            physical_area = enclosed | inner_stroke
            for channel, seed in zip(requested_channels, seeds):
                masks[channel] = physical_area & seed
                if not masks[channel].any():
                    raise ValueError(f"Empty CAD intersection for {channel}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **masks)
    return masks
