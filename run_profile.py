from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cad_mask import build_cad_masks
from profile_store import (
    load_profile,
    profile_path,
    project_root,
    resolve_profile_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FLUID-Space profile")
    parser.add_argument("--profile", default="current_baseline")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def generate_compatibility_files(profile_name: str) -> tuple[dict[str, Any], Path, Path]:
    root = project_root()
    profile = load_profile(profile_name, require_complete=True)
    video_path = resolve_profile_video(profile_name, profile)
    if not video_path.is_file():
        raise FileNotFoundError(f"Profile video not found: {video_path}")
    profile_dir = profile_path(profile_name).parent
    generated = profile_dir / ".generated"
    generated.mkdir(parents=True, exist_ok=True)
    frame_start = int(profile["frame"]["start"])
    frame_end = int(profile["frame"]["end"])
    frame_step = int(profile["frame"]["step"])
    result_channels = list(profile["settings"]["result_channels"])
    fps = float(profile.get("source_video", {}).get("fps", 30.0) or 30.0)
    roi_path = generated / "roi.json"
    cad_masks_path = generated / "cad_masks.npz"
    roi_payload = {
        "schema_version": 4,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_step": frame_step,
            "frame_width": int(profile["frame"]["width"]),
            "frame_height": int(profile["frame"]["height"]),
            "fps": fps,
        },
        "channels": result_channels,
        "rois": profile["rois"],
        "zones": profile["interest_zones"],
        "calibration_rois": profile["calibration_rois"],
        "exclusion_islands": [],
        "notes": {
            "geometry_policy": "CAD authoritative; channel ROI and channel Interest Zone only remove CAD pixels.",
            "generated_from_profile": profile["name"],
            "bubble_policy": "Bubble annotations are retained in the profile but not used for time-series analysis.",
        },
    }
    write_json(roi_path, roi_payload)
    build_cad_masks(root / "assets" / "cad_pair_lines.npz", roi_payload, cad_masks_path)
    return profile, roi_path, cad_masks_path


def command_for_profile(profile_name: str, check: bool = False) -> list[str]:
    root = project_root()
    profile, roi_path, cad_masks_path = generate_compatibility_files(profile_name)
    video_path = resolve_profile_video(profile_name, profile)
    command = [
        sys.executable,
        str(root / "time_series_pipeline.py"),
        "--video", str(video_path),
        "--project", str(roi_path),
        "--start-frame", str(profile["frame"]["start"]),
        "--end-frame", str(profile["frame"]["end"]),
        "--frame-step", str(profile["frame"]["step"]),
        "--cad-roi-masks", str(cad_masks_path),
        "--cad-placement-config", str(root / "assets" / "cad_placement.json"),
        "--output-root", str(profile_path(profile_name).parent / "runs"),
    ]
    if check:
        command.append("--check")
    return command


def main() -> int:
    args = parse_args()
    command = command_for_profile(args.profile, args.check)
    completed = subprocess.run(command, cwd=project_root())
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
