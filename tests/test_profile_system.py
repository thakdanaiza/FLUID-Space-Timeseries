from __future__ import annotations

import json
import argparse
import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import profile_store
import numpy as np
from cad_mask import build_cad_masks
from phase_analysis_core import refine_bubble_boundaries
from run_profile import command_for_profile, generate_compatibility_files
from time_series_pipeline import circular_mean, run, sampled_frame_indices


class ProfileSystemTests(unittest.TestCase):
    def test_baseline_is_complete_and_contains_imported_bubbles(self) -> None:
        profile = profile_store.load_profile("current_baseline", require_complete=True)
        self.assertEqual(set(profile["interest_zones"]), set(profile_store.ZONES))
        self.assertEqual(set(profile["rois"]), set(profile_store.CHANNELS))
        self.assertGreater(sum(len(profile["bubbles"][name]) for name in profile_store.CHANNELS), 0)

    def test_generated_roi_has_no_manual_islands(self) -> None:
        _, roi_path, cad_path = generate_compatibility_files("current_baseline")
        roi = json.loads(roi_path.read_text(encoding="utf-8"))
        self.assertEqual(roi["exclusion_islands"], [])
        self.assertEqual(set(roi["zones"]), set(profile_store.CHANNELS))
        self.assertTrue(cad_path.exists())

    def test_profile_video_is_bound_and_used_by_pipeline(self) -> None:
        profile = profile_store.load_profile("current_baseline", require_complete=True)
        video_path = profile_store.resolve_profile_video("current_baseline", profile)
        self.assertTrue(video_path.is_file())

        _, roi_path, _ = generate_compatibility_files("current_baseline")
        roi = json.loads(roi_path.read_text(encoding="utf-8"))
        self.assertEqual(Path(roi["source"]["video_path"]), video_path)

        command = command_for_profile("current_baseline")
        self.assertEqual(Path(command[command.index("--video") + 1]), video_path)
        self.assertEqual(Path(command[1]).name, "time_series_pipeline.py")
        self.assertEqual(
            int(command[command.index("--start-frame") + 1]), profile["frame"]["start"]
        )
        self.assertEqual(
            int(command[command.index("--end-frame") + 1]), profile["frame"]["end"]
        )
        self.assertEqual(
            int(command[command.index("--frame-step") + 1]), profile["frame"]["step"]
        )
        self.assertNotIn("--analysis-project", command)

    def test_legacy_profile_migrates_to_time_series_schema(self) -> None:
        current = json.loads(
            (ROOT / "profiles" / "current_baseline" / "profile.json").read_text(
                encoding="utf-8"
            )
        )
        legacy = copy.deepcopy(current)
        if int(legacy["schema_version"]) == 2:
            legacy["schema_version"] = 1
            legacy["frame"] = {
                "index": current["frame"]["start"],
                "width": current["frame"]["width"],
                "height": current["frame"]["height"],
            }
            legacy["interest_zones"] = {
                "ZONE-CH1": copy.deepcopy(current["interest_zones"]["CH1-1"]),
                "ZONE-CH2": copy.deepcopy(current["interest_zones"]["CH2-1"]),
            }
        migrated = profile_store.validate_profile(legacy, require_complete=True)
        self.assertEqual(migrated["schema_version"], 2)
        self.assertEqual(migrated["frame"]["start"], legacy["frame"]["index"])
        self.assertEqual(migrated["frame"]["end"], legacy["frame"]["index"])
        self.assertEqual(migrated["frame"]["step"], 1)
        self.assertEqual(migrated["interest_zones"]["CH1-1"], legacy["interest_zones"]["ZONE-CH1"])
        self.assertEqual(migrated["interest_zones"]["CH2-2"], legacy["interest_zones"]["ZONE-CH2"])

    def test_frame_sampling_is_inclusive_and_uniform(self) -> None:
        self.assertEqual(sampled_frame_indices(10, 20, 4, 30), [10, 14, 18])
        with self.assertRaises(ValueError):
            sampled_frame_indices(20, 10, 1, 30)
        with self.assertRaises(ValueError):
            sampled_frame_indices(0, 30, 1, 30)

    def test_result_channel_selection_is_validated(self) -> None:
        profile = profile_store.load_profile("current_baseline", require_complete=True)
        profile["settings"]["result_channels"] = ["CH2-1"]
        for channel in ("CH1-1", "CH1-2", "CH2-2"):
            profile["interest_zones"][channel] = []
            profile["rois"][channel] = []
        profile["calibration_rois"]["CAL-CH1"] = []
        validated = profile_store.validate_profile(profile, require_complete=True)
        self.assertEqual(validated["settings"]["result_channels"], ["CH2-1"])
        profile["settings"]["result_channels"] = []
        with self.assertRaises(ValueError):
            profile_store.validate_profile(profile, require_complete=True)

    def test_circular_calibration_average_handles_wraparound(self) -> None:
        mean = circular_mean([0.49, -0.49])
        self.assertAlmostEqual(abs(mean), 0.5, places=6)

    def test_time_series_run_writes_only_graph_csv_and_json(self) -> None:
        profile, roi_path, _ = generate_compatibility_files("current_baseline")
        video_path = profile_store.resolve_profile_video("current_baseline", profile)
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            roi_payload = json.loads(roi_path.read_text(encoding="utf-8"))
            roi_payload["channels"] = ["CH1-2"]
            selected_roi_path = temporary_path / "selected_roi.json"
            selected_roi_path.write_text(json.dumps(roi_payload), encoding="utf-8")
            selected_cad_path = temporary_path / "selected_cad_masks.npz"
            build_cad_masks(
                ROOT / "assets" / "cad_pair_lines.npz",
                roi_payload,
                selected_cad_path,
            )
            args = argparse.Namespace(
                video=str(video_path),
                project=str(selected_roi_path),
                start_frame=profile["frame"]["start"],
                end_frame=profile["frame"]["end"],
                frame_step=profile["frame"]["step"],
                cad_roi_masks=str(selected_cad_path),
                cad_placement_config=str(ROOT / "assets" / "cad_placement.json"),
                output_root=temporary,
                check=False,
            )
            run_dir = run(args)
            self.assertIsNotNone(run_dir)
            files = {
                path.relative_to(run_dir).as_posix()
                for path in run_dir.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                files,
                {"graphs/phase_time_series.png", "time_series.csv", "summary.json"},
            )
            rows = (run_dir / "time_series.csv").read_text(encoding="utf-8")
            self.assertIn(",CH1-2,", rows)
            self.assertNotIn(",CH1-1,", rows)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["result_channels"], ["CH1-2"])

    def test_profile_slug_rejects_empty_names(self) -> None:
        with self.assertRaises(ValueError):
            profile_store.slugify(" !!! ")

    def test_bubble_refinement_never_restores_manually_excluded_pixels(self) -> None:
        frame = np.full((80, 120, 3), 128, dtype=np.uint8)
        wet = np.ones((80, 120), dtype=bool)
        exclusion = {
            "id": "CH1-2-BUBBLE-TEST",
            "kind": "bubble",
            "points": [[20, 15], [94, 18], [103, 39], [87, 61], [30, 66], [15, 43]],
        }
        refinement = refine_bubble_boundaries(
            frame,
            wet,
            [exclusion],
            boundary_width=3,
            smooth_radius=3,
        )
        restored = refinement.drawn_mask & ~refinement.proposed_mask
        self.assertEqual(int(restored.sum()), 0)


if __name__ == "__main__":
    unittest.main()
