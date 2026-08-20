from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from profile_store import (
    CHANNELS,
    VIDEO_OUTPUTS,
    copy_video_into_profile,
    duplicate_profile,
    list_profiles,
    load_profile,
    profile_completeness,
    profile_path,
    project_root,
    resolve_profile_video,
    save_profile,
)


ACTIONS = (
    ("Interest Zone", "zone"),
    ("Channel ROI", "roi"),
    ("Bubble", "bubble"),
)

COLORS = {
    "zone": "#44df7b",
    "roi": "#2ed5ff",
    "bubble": "#ff4f6d",
    "draft": "#ffd166",
    "cad": "#101010",
}


class FluidSpaceApp:
    def __init__(self, root: tk.Tk, initial_profile: str) -> None:
        self.root = root
        self.root.title("FLUID-Space · CAD-authoritative time series")
        self.root.geometry("1440x820")
        self.root.minsize(1050, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.profile: dict[str, Any] = {}
        self.profile_name = ""
        self.dirty = False
        self.draft: list[list[float]] = []
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.fit_scale = 1.0
        self._pan_anchor: tuple[float, float] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._run_process: subprocess.Popen[str] | None = None

        self.frame_image = Image.open(project_root() / "assets" / "frame_0120.png").convert("RGB")
        vector_payload = json.loads((project_root() / "assets" / "cad_pair_vector_lines.json").read_text(encoding="utf-8"))
        self.cad_paths: dict[str, list[list[list[float]]]] = vector_payload["groups"]

        self.profile_var = tk.StringVar()
        self.video_var = tk.StringVar(value="Video: loading…")
        self.frame_start_var = tk.StringVar(value="120")
        self.frame_end_var = tk.StringVar(value="120")
        self.frame_step_var = tk.StringVar(value="1")
        self.active_channel = tk.StringVar(value=CHANNELS[0])
        self.result_channel_vars = {
            channel: tk.BooleanVar(value=True) for channel in CHANNELS
        }
        self.video_output_vars = {
            output: tk.BooleanVar(value=False) for output in VIDEO_OUTPUTS
        }
        self.cad_visible = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.geometry_status_var = tk.StringVar()
        self._build_ui()
        self._bind_shortcuts()
        self.refresh_profiles(initial_profile)
        self.root.after(80, self.fit_view)

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("Step.TLabel", font=("Segoe UI", 10, "bold"))

        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Profile", style="Step.TLabel").pack(side="left")
        self.profile_combo = ttk.Combobox(header, textvariable=self.profile_var, state="readonly", width=28)
        self.profile_combo.pack(side="left", padx=(8, 6))
        self.profile_combo.bind("<<ComboboxSelected>>", self._profile_selected)
        ttk.Button(header, text="Duplicate as…", command=self.duplicate_current).pack(side="left", padx=3)
        ttk.Button(header, text="Save", style="Primary.TButton", command=self.save).pack(side="left", padx=(12, 3))
        self.run_button = ttk.Button(header, text="Run result", style="Primary.TButton", command=self.run_result)
        self.run_button.pack(side="left", padx=3)
        ttk.Checkbutton(header, text="CAD overlay", variable=self.cad_visible, command=self.redraw).pack(side="right")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        sidebar = ttk.Frame(body, width=270, padding=(0, 4, 10, 4))
        work = ttk.Frame(body)
        body.add(sidebar, weight=0)
        body.add(work, weight=1)

        ttk.Label(sidebar, text="1  Source video", style="Step.TLabel").pack(anchor="w", pady=(2, 4))
        ttk.Label(sidebar, textvariable=self.video_var, justify="left", wraplength=250).pack(anchor="w", pady=(0, 5))
        ttk.Button(sidebar, text="Select video…", command=self.select_video).pack(fill="x")
        frame_row = ttk.Frame(sidebar)
        frame_row.pack(fill="x", pady=(6, 0))
        ttk.Label(frame_row, text="Start").pack(side="left")
        self.frame_spinbox = ttk.Spinbox(frame_row, from_=0, to=999999, textvariable=self.frame_start_var, width=8)
        self.frame_spinbox.pack(side="left", padx=(6, 4))
        ttk.Button(frame_row, text="Load Start", command=self.load_selected_frame).pack(
            side="left", expand=True, fill="x"
        )
        range_row = ttk.Frame(sidebar)
        range_row.pack(fill="x", pady=(4, 0))
        ttk.Label(range_row, text="End").pack(side="left")
        self.frame_end_spinbox = ttk.Spinbox(
            range_row, from_=0, to=999999, textvariable=self.frame_end_var, width=8
        )
        self.frame_end_spinbox.pack(side="left", padx=(11, 4))
        ttk.Button(range_row, text="Load End", command=self.load_end_frame).pack(
            side="left", expand=True, fill="x"
        )
        step_row = ttk.Frame(sidebar)
        step_row.pack(fill="x", pady=(4, 0))
        ttk.Label(step_row, text="Step").pack(side="left")
        self.frame_step_spinbox = ttk.Spinbox(
            step_row, from_=1, to=999999, textvariable=self.frame_step_var, width=8
        )
        self.frame_step_spinbox.pack(side="left", padx=(8, 0))

        ttk.Separator(sidebar).pack(fill="x", pady=12)
        ttk.Label(sidebar, text="2  Select channel", style="Step.TLabel").pack(anchor="w", pady=(2, 6))
        channel_frame = ttk.Frame(sidebar)
        channel_frame.pack(fill="x")
        for index, channel in enumerate(CHANNELS):
            ttk.Radiobutton(
                channel_frame,
                text=channel,
                value=channel,
                variable=self.active_channel,
                command=self.change_channel,
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 18), pady=2)
        ttk.Label(sidebar, text="Result channels").pack(anchor="w", pady=(7, 2))
        result_channel_frame = ttk.Frame(sidebar)
        result_channel_frame.pack(fill="x")
        for index, channel in enumerate(CHANNELS):
            ttk.Checkbutton(
                result_channel_frame,
                text=channel,
                variable=self.result_channel_vars[channel],
                command=self.result_channels_changed,
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 14), pady=1)
        ttk.Label(sidebar, text="Video outputs").pack(anchor="w", pady=(7, 2))
        video_output_frame = ttk.Frame(sidebar)
        video_output_frame.pack(fill="x")
        for output in VIDEO_OUTPUTS:
            ttk.Checkbutton(
                video_output_frame,
                text=output.title(),
                variable=self.video_output_vars[output],
                command=self.video_outputs_changed,
            ).pack(side="left", padx=(0, 18))

        ttk.Separator(sidebar).pack(fill="x", pady=12)
        ttk.Label(sidebar, text="3  Select drawing", style="Step.TLabel").pack(anchor="w", pady=(0, 6))
        self.target_list = tk.Listbox(sidebar, height=len(ACTIONS), exportselection=False, activestyle="none", font=("Segoe UI", 10))
        for label, _ in ACTIONS:
            self.target_list.insert("end", label)
        self.target_list.selection_set(0)
        self.target_list.bind("<<ListboxSelect>>", lambda _event: self.change_target())
        self.target_list.pack(fill="x")

        ttk.Separator(sidebar).pack(fill="x", pady=12)
        ttk.Label(sidebar, text="4  Draw on image", style="Step.TLabel").pack(anchor="w")
        instructions = (
            "Left click: add a point\n"
            "Right click / Enter: close polygon\n"
            "Mouse wheel: zoom\n"
            "Middle-button drag: pan\n"
            "Esc: cancel the current draft"
        )
        ttk.Label(sidebar, text=instructions, justify="left").pack(anchor="w", pady=(6, 8))
        actions = ttk.Frame(sidebar)
        actions.pack(fill="x")
        ttk.Button(actions, text="Undo point", command=self.undo_point).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(actions, text="Commit", command=self.commit_draft).pack(side="left", expand=True, fill="x", padx=(3, 0))
        ttk.Button(sidebar, text="Clear current", command=self.clear_current).pack(fill="x", pady=(6, 0))
        ttk.Button(sidebar, text="Fit image", command=self.fit_view).pack(fill="x", pady=(6, 0))

        ttk.Separator(sidebar).pack(fill="x", pady=12)
        ttk.Label(sidebar, text="Profile completeness", style="Step.TLabel").pack(anchor="w")
        ttk.Label(sidebar, textvariable=self.geometry_status_var, justify="left", wraplength=250).pack(anchor="w", pady=(5, 0))
        ttk.Label(
            sidebar,
            text="CAD and calibration are locked assets.\nAll internal islands come from CAD only.",
            foreground="#4a6670",
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

        self.canvas = tk.Canvas(work, background="#d9dde1", highlightthickness=1, highlightbackground="#a8b0b7")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", lambda _event: self.commit_draft())
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<ButtonPress-2>", self.pan_start)
        self.canvas.bind("<B2-Motion>", self.pan_move)
        self.canvas.bind("<ButtonRelease-2>", lambda _event: setattr(self, "_pan_anchor", None))

        footer = ttk.Frame(self.root, padding=(12, 3, 12, 9))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=180)
        self.progress.pack(side="right")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Return>", lambda _event: self.commit_draft())
        self.root.bind("<Escape>", lambda _event: self.cancel_draft())
        self.root.bind("<Control-s>", lambda _event: self.save())
        self.root.bind("<Control-z>", lambda _event: self.undo_point())

    def refresh_profiles(self, select: str | None = None) -> None:
        names = list_profiles()
        self.profile_combo["values"] = names
        chosen = select if select in names else (names[0] if names else "")
        if not chosen:
            raise RuntimeError("No profiles found. Add a profile under profiles/<name>/profile.json.")
        self.load(chosen)

    def load(self, name: str) -> None:
        self.profile = load_profile(name, require_complete=False)
        self.profile_name = name
        self.profile_var.set(name)
        frame = self.profile["frame"]
        self.frame_start_var.set(str(frame["start"]))
        self.frame_end_var.set(str(frame["end"]))
        self.frame_step_var.set(str(frame["step"]))
        selected_results = set(self.profile["settings"]["result_channels"])
        for channel, variable in self.result_channel_vars.items():
            variable.set(channel in selected_results)
        selected_video_outputs = set(self.profile["settings"]["video_outputs"])
        for output, variable in self.video_output_vars.items():
            variable.set(output in selected_video_outputs)
        self.draft = []
        self.dirty = False
        self._load_bound_video(show_error=True)
        self.status_var.set(f"Loaded profile: {name}")
        self.update_completeness()
        self.redraw()

    def _profile_selected(self, _event: tk.Event[Any]) -> None:
        selected = self.profile_var.get()
        if selected == self.profile_name:
            return
        if self.dirty and not messagebox.askyesno("Unsaved changes", "Discard unsaved edits and switch profile?"):
            self.profile_var.set(self.profile_name)
            return
        self.load(selected)

    def duplicate_current(self) -> None:
        name = simpledialog.askstring("Duplicate profile", "New profile name:", parent=self.root)
        if not name:
            return
        try:
            path = duplicate_profile(self.profile_name, name)
        except Exception as exc:
            messagebox.showerror("Cannot duplicate profile", str(exc))
            return
        self.refresh_profiles(path.parent.name)
        self.status_var.set(f"Created profile: {path.parent.name}")

    def save(self) -> bool:
        if self.draft and not self.commit_draft():
            return False
        try:
            frame_start = int(self.frame_start_var.get().strip())
            frame_end = int(self.frame_end_var.get().strip())
            frame_step = int(self.frame_step_var.get().strip())
            if frame_start < 0 or frame_end < frame_start or frame_step < 1:
                raise ValueError("Use Start >= 0, End >= Start, and Step >= 1")
            frame_count = int(self.profile.get("source_video", {}).get("frame_count", 0))
            if frame_count and frame_end >= frame_count:
                raise ValueError(
                    f"End frame {frame_end} is outside this video (last frame: {frame_count - 1})"
                )
            self.profile["frame"].update(
                {"start": frame_start, "end": frame_end, "step": frame_step}
            )
            result_channels = [
                channel for channel in CHANNELS if self.result_channel_vars[channel].get()
            ]
            if not result_channels:
                raise ValueError("Select at least one Result channel")
            self.profile.setdefault("settings", {})["result_channels"] = result_channels
            self.profile["settings"]["video_outputs"] = [
                output for output in VIDEO_OUTPUTS if self.video_output_vars[output].get()
            ]
            path = save_profile(self.profile, require_complete=False)
            self.profile = load_profile(path.parent.name, require_complete=False)
            self.dirty = False
            self.status_var.set(f"Saved: {path}")
            self.update_completeness()
            return True
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return False

    def _read_video_frame(
        self,
        path: Path,
        frame_index: int,
        clamp_index: bool = False,
    ) -> tuple[Image.Image, dict[str, Any]]:
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY
            capture = cv2.VideoCapture(str(path), backend)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
            if frame_index < 0:
                raise ValueError("Frame index cannot be negative")
            if frame_count and frame_index >= frame_count:
                if clamp_index:
                    frame_index = max(0, frame_count - 1)
                else:
                    raise ValueError(
                        f"Frame {frame_index} is outside this video (last frame: {frame_count - 1})"
                    )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read frame {frame_index} from {path.name}")
        height, width = frame.shape[:2]
        expected = (int(self.profile["frame"]["width"]), int(self.profile["frame"]["height"]))
        if (width, height) != expected:
            raise ValueError(
                f"Video resolution is {width}x{height}; this CAD profile requires "
                f"{expected[0]}x{expected[1]}."
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        metadata = {
            "frame_index": frame_index,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
        }
        return Image.fromarray(rgb), metadata

    def _load_bound_video(self, show_error: bool) -> bool:
        try:
            video_path = resolve_profile_video(self.profile_name, self.profile)
            image, metadata = self._read_video_frame(
                video_path,
                int(self.profile["frame"]["start"]),
            )
        except Exception as exc:
            self.video_var.set(f"Video unavailable: {exc}")
            self.frame_image = Image.open(project_root() / "assets" / "frame_0120.png").convert("RGB")
            if show_error:
                messagebox.showerror("Profile video unavailable", str(exc))
            return False
        self.frame_image = image
        self.frame_start_var.set(str(metadata["frame_index"]))
        maximum = max(0, metadata["frame_count"] - 1)
        self.frame_spinbox.configure(to=maximum)
        self.frame_end_spinbox.configure(to=maximum)
        self.frame_step_spinbox.configure(to=max(1, metadata["frame_count"]))
        source = self.profile.setdefault("source_video", {})
        display_name = str(source.get("name") or video_path.name)
        source.update(
            {
                "fps": metadata["fps"],
                "frame_count": metadata["frame_count"],
            }
        )
        self.video_var.set(
            f"{display_name}\n"
            f"{metadata['width']}×{metadata['height']} · {metadata['frame_count']} frames · "
            f"{metadata['fps']:.3g} fps"
        )
        return True

    def select_video(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title=f"Select video for profile {self.profile_name}",
            filetypes=(
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                ("All files", "*.*"),
            ),
        )
        if not selected:
            return
        if self.draft:
            if not messagebox.askyesno(
                "Cancel current draft?",
                "Selecting a video will cancel the unfinished polygon. Continue?",
            ):
                return
            self.draft = []
        source_path = Path(selected)
        try:
            image, metadata = self._read_video_frame(
                source_path,
                int(self.profile["frame"]["start"]),
                clamp_index=True,
            )
            maximum = max(0, metadata["frame_count"] - 1)
            start = int(metadata["frame_index"])
            end = min(max(start, int(self.profile["frame"]["end"])), maximum)
            step = max(1, int(self.profile["frame"]["step"]))
            destination, relative = copy_video_into_profile(self.profile_name, source_path)
            self.profile["source_video"] = {
                "path": relative,
                "name": source_path.name,
                "stored_name": destination.name,
                "fps": metadata["fps"],
                "frame_count": metadata["frame_count"],
            }
            self.profile["frame"].update({"start": start, "end": end, "step": step})
            self.frame_image = image
            self.frame_start_var.set(str(start))
            self.frame_end_var.set(str(end))
            self.frame_step_var.set(str(step))
            self.frame_spinbox.configure(to=maximum)
            self.frame_end_spinbox.configure(to=maximum)
            self.frame_step_spinbox.configure(to=max(1, metadata["frame_count"]))
            self.video_var.set(
                f"{source_path.name}\n"
                f"{metadata['width']}×{metadata['height']} · {metadata['frame_count']} frames · "
                f"{metadata['fps']:.3g} fps"
            )
            self.dirty = True
            if not self.save():
                return
            self.fit_view()
            self.status_var.set(f"Video bound to profile {self.profile_name}: {source_path.name}")
        except Exception as exc:
            messagebox.showerror("Cannot select video", str(exc))

    def load_selected_frame(self) -> None:
        self._load_range_frame("start")

    def load_end_frame(self) -> None:
        self._load_range_frame("end")

    def _load_range_frame(self, position: str) -> None:
        try:
            frame_start = int(self.frame_start_var.get().strip())
            frame_end = int(self.frame_end_var.get().strip())
            frame_step = int(self.frame_step_var.get().strip())
            if frame_start < 0 or frame_end < frame_start or frame_step < 1:
                raise ValueError("Use Start >= 0, End >= Start, and Step >= 1")
            video_path = resolve_profile_video(self.profile_name, self.profile)
            preview_index = frame_start if position == "start" else frame_end
            image, metadata = self._read_video_frame(video_path, preview_index)
            if metadata["frame_count"] and frame_end >= metadata["frame_count"]:
                raise ValueError(
                    f"End frame {frame_end} is outside this video "
                    f"(last frame: {metadata['frame_count'] - 1})"
                )
            self.profile["frame"].update(
                {"start": frame_start, "end": frame_end, "step": frame_step}
            )
            self.profile.setdefault("source_video", {}).update(
                {"fps": metadata["fps"], "frame_count": metadata["frame_count"]}
            )
            self.frame_image = image
            self.dirty = True
            if not self.save():
                return
            self.fit_view()
            sampled = range(frame_start, frame_end + 1, frame_step)
            sample_count = len(sampled)
            self.status_var.set(
                f"Loaded {position} frame {preview_index}; "
                f"{sample_count} samples through frame {sampled[-1]}"
            )
        except Exception as exc:
            messagebox.showerror("Cannot load frame", str(exc))

    def current_target(self) -> tuple[str, str]:
        selection = self.target_list.curselection()
        index = selection[0] if selection else 0
        _, kind = ACTIONS[index]
        channel = self.active_channel.get()
        name = channel
        return kind, name

    def change_channel(self) -> None:
        if self.draft:
            self.draft = []
            self.status_var.set("Draft cancelled because the active channel changed")
        self.update_completeness()
        self.redraw()

    def result_channels_changed(self) -> None:
        selected = [
            channel for channel in CHANNELS if self.result_channel_vars[channel].get()
        ]
        self.profile.setdefault("settings", {})["result_channels"] = selected
        self.dirty = True
        self.status_var.set(
            "Result channels: " + (", ".join(selected) if selected else "select at least one")
        )
        self.update_completeness()

    def video_outputs_changed(self) -> None:
        selected = [
            output for output in VIDEO_OUTPUTS if self.video_output_vars[output].get()
        ]
        self.profile.setdefault("settings", {})["video_outputs"] = selected
        self.dirty = True
        self.status_var.set(
            "Video outputs: " + (", ".join(name.title() for name in selected) if selected else "None")
        )
        self.update_completeness()

    def change_target(self) -> None:
        if self.draft:
            self.draft = []
            self.status_var.set("Draft cancelled because the drawing target changed")
        self.redraw()

    def add_point(self, event: tk.Event[Any]) -> None:
        point = self.canvas_to_image(float(event.x), float(event.y))
        if point is None:
            return
        self.draft.append([round(point[0], 3), round(point[1], 3)])
        self.status_var.set(f"Draft: {len(self.draft)} points")
        self.redraw()

    def undo_point(self) -> None:
        if self.draft:
            self.draft.pop()
            self.status_var.set(f"Draft: {len(self.draft)} points")
            self.redraw()

    def cancel_draft(self) -> None:
        self.draft = []
        self.status_var.set("Draft cancelled")
        self.redraw()

    def commit_draft(self) -> bool:
        if not self.draft:
            return True
        if len(self.draft) < 3:
            messagebox.showwarning("Polygon incomplete", "A polygon needs at least 3 points")
            return False
        kind, name = self.current_target()
        if kind == "zone":
            self.profile["interest_zones"][name] = copy.deepcopy(self.draft)
        elif kind == "roi":
            self.profile["rois"][name] = copy.deepcopy(self.draft)
        else:
            items = self.profile["bubbles"][name]
            next_id = 1
            existing = {item["id"] for item in items}
            while f"{name}-BUBBLE-{next_id:03d}" in existing:
                next_id += 1
            items.append({"id": f"{name}-BUBBLE-{next_id:03d}", "points": copy.deepcopy(self.draft)})
        self.draft = []
        self.dirty = True
        self.status_var.set(f"Updated {name} — press Save when ready")
        self.update_completeness()
        self.redraw()
        return True

    def clear_current(self) -> None:
        kind, name = self.current_target()
        if kind == "bubble":
            items = self.profile["bubbles"][name]
            if not items:
                return
            if messagebox.askyesno("Clear bubble", f"Remove the latest bubble from {name}?"):
                items.pop()
        else:
            if messagebox.askyesno("Clear geometry", f"Clear {name} from this profile?"):
                key = "interest_zones" if kind == "zone" else "rois"
                self.profile[key][name] = []
        self.draft = []
        self.dirty = True
        self.update_completeness()
        self.redraw()

    def update_completeness(self) -> None:
        complete, missing = profile_completeness(self.profile)
        bubble_count = sum(len(self.profile.get("bubbles", {}).get(channel, [])) for channel in CHANNELS)
        active = self.active_channel.get()
        active_zone = active
        active_ready = (
            len(self.profile.get("interest_zones", {}).get(active_zone, [])) >= 3
            and len(self.profile.get("rois", {}).get(active, [])) >= 3
        )
        active_bubbles = len(self.profile.get("bubbles", {}).get(active, []))
        result_channels = [
            channel for channel in CHANNELS if self.result_channel_vars[channel].get()
        ]
        result_text = ", ".join(result_channels) if result_channels else "None selected"
        video_outputs = [
            output.title() for output in VIDEO_OUTPUTS if self.video_output_vars[output].get()
        ]
        video_text = ", ".join(video_outputs) if video_outputs else "None (images only)"
        if complete:
            text = (
                f"Active: {active}\n"
                f"{'✓' if active_ready else '○'} Zone + ROI ready\n"
                f"Bubbles in channel: {active_bubbles}\n\n"
                f"Selected geometry: Zones {len(result_channels)}/{len(result_channels)} · "
                f"ROIs {len(result_channels)}/{len(result_channels)}\n"
                f"Results: {result_text}\n"
                f"Videos: {video_text}\n"
                f"Total bubbles: {bubble_count} (not used in time series)"
            )
        else:
            text = (
                f"Active: {active}\n"
                f"Bubbles in channel: {active_bubbles}\n\n"
                "Missing:\n"
                + "\n".join(f"• {item}" for item in missing)
                + f"\nResults: {result_text}"
                + f"\nVideos: {video_text}"
                + f"\nTotal bubbles: {bubble_count} (not used in time series)"
            )
        self.geometry_status_var.set(text)

    def fit_view(self) -> None:
        width = max(100, self.canvas.winfo_width())
        height = max(100, self.canvas.winfo_height())
        self.fit_scale = min((width - 30) / self.frame_image.width, (height - 30) / self.frame_image.height)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw()

    def transform(self) -> tuple[float, float, float]:
        scale = max(0.05, self.fit_scale * self.zoom)
        image_w = self.frame_image.width * scale
        image_h = self.frame_image.height * scale
        left = (self.canvas.winfo_width() - image_w) / 2 + self.pan_x
        top = (self.canvas.winfo_height() - image_h) / 2 + self.pan_y
        return scale, left, top

    def image_to_canvas(self, point: list[float]) -> tuple[float, float]:
        scale, left, top = self.transform()
        return left + point[0] * scale, top + point[1] * scale

    def canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        scale, left, top = self.transform()
        ix, iy = (x - left) / scale, (y - top) / scale
        if 0 <= ix < self.frame_image.width and 0 <= iy < self.frame_image.height:
            return ix, iy
        return None

    def on_wheel(self, event: tk.Event[Any]) -> None:
        before = self.canvas_to_image(float(event.x), float(event.y))
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.zoom = min(8.0, max(0.35, self.zoom * factor))
        if before is not None:
            scale, left, top = self.transform()
            target_x = left + before[0] * scale
            target_y = top + before[1] * scale
            self.pan_x += float(event.x) - target_x
            self.pan_y += float(event.y) - target_y
        self.redraw()

    def pan_start(self, event: tk.Event[Any]) -> None:
        self._pan_anchor = float(event.x) - self.pan_x, float(event.y) - self.pan_y

    def pan_move(self, event: tk.Event[Any]) -> None:
        if self._pan_anchor is None:
            return
        self.pan_x = float(event.x) - self._pan_anchor[0]
        self.pan_y = float(event.y) - self._pan_anchor[1]
        self.redraw()

    def _draw_polygon(self, points: list[list[float]], color: str, width: int, dash: tuple[int, int] | None = None) -> None:
        if not points:
            return
        coords = [coordinate for point in points for coordinate in self.image_to_canvas(point)]
        if len(points) >= 2:
            if len(points) >= 3:
                coords += list(self.image_to_canvas(points[0]))
            self.canvas.create_line(*coords, fill=color, width=width, dash=dash, joinstyle="round")
        for point in points:
            x, y = self.image_to_canvas(point)
            radius = 3
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="")

    def _channel_rectangle(self, channel: str) -> tuple[float, float, float, float]:
        group = channel.split("-", 1)[0]
        channel_pair = (f"{group}-1", f"{group}-2")
        zone = self.profile["interest_zones"].get(channel, [])
        xs = [float(point[0]) for point in zone] or [0.0, float(self.frame_image.width - 1)]

        def center_y(name: str, fallback: float) -> float:
            points = self.profile["rois"].get(name, [])
            return sum(float(point[1]) for point in points) / len(points) if points else fallback

        first_center = center_y(channel_pair[0], self.frame_image.height * 0.25)
        second_center = center_y(channel_pair[1], self.frame_image.height * 0.75)
        separator = (first_center + second_center) / 2.0
        xmin, xmax = min(xs), max(xs)
        first_is_upper = first_center <= second_center
        if (channel == channel_pair[0]) == first_is_upper:
            return xmin, 0.0, xmax, separator
        return xmin, separator, xmax, float(self.frame_image.height - 1)

    @staticmethod
    def _clip_segment(
        start: list[float],
        end: list[float],
        rectangle: tuple[float, float, float, float],
    ) -> tuple[list[float], list[float]] | None:
        xmin, ymin, xmax, ymax = rectangle
        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        directions = (-dx, dx, -dy, dy)
        distances = (float(start[0]) - xmin, xmax - float(start[0]), float(start[1]) - ymin, ymax - float(start[1]))
        lower, upper = 0.0, 1.0
        for direction, distance in zip(directions, distances):
            if abs(direction) < 1e-12:
                if distance < 0:
                    return None
                continue
            ratio = distance / direction
            if direction < 0:
                lower = max(lower, ratio)
            else:
                upper = min(upper, ratio)
            if lower > upper:
                return None
        return (
            [float(start[0]) + lower * dx, float(start[1]) + lower * dy],
            [float(start[0]) + upper * dx, float(start[1]) + upper * dy],
        )

    def _draw_active_cad(self, channel: str, scale: float) -> None:
        group = channel.split("-", 1)[0]
        rectangle = self._channel_rectangle(channel)
        for path in self.cad_paths[group]:
            if len(path) < 2:
                continue
            stride = max(1, len(path) // 260)
            shown = path[::stride]
            if shown[-1] != path[-1]:
                shown.append(path[-1])
            runs: list[list[list[float]]] = []
            current: list[list[float]] = []
            for start, end in zip(shown[:-1], shown[1:]):
                clipped = self._clip_segment(start, end, rectangle)
                if clipped is None:
                    if len(current) >= 2:
                        runs.append(current)
                    current = []
                    continue
                clipped_start, clipped_end = clipped
                if current and abs(current[-1][0] - clipped_start[0]) < 0.5 and abs(current[-1][1] - clipped_start[1]) < 0.5:
                    current.append(clipped_end)
                else:
                    if len(current) >= 2:
                        runs.append(current)
                    current = [clipped_start, clipped_end]
            if len(current) >= 2:
                runs.append(current)
            for run in runs:
                coords = [coordinate for point in run for coordinate in self.image_to_canvas(point)]
                self.canvas.create_line(
                    *coords,
                    fill=COLORS["cad"],
                    width=max(1, round(1.4 * scale)),
                    joinstyle="round",
                    capstyle="round",
                )

    def redraw(self) -> None:
        if not hasattr(self, "canvas") or not self.profile:
            return
        self.canvas.delete("all")
        scale, left, top = self.transform()
        size = (max(1, round(self.frame_image.width * scale)), max(1, round(self.frame_image.height * scale)))
        resized = self.frame_image.resize(size, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(left, top, image=self._photo, anchor="nw")

        active_channel = self.active_channel.get()
        kind, active_name = self.current_target()
        active_zone = active_channel
        if self.cad_visible.get():
            self._draw_active_cad(active_channel, scale)
        self._draw_polygon(
            self.profile["interest_zones"].get(active_zone, []),
            COLORS["zone"],
            3 if kind == "zone" else 1,
            (8, 4),
        )
        self._draw_polygon(
            self.profile["rois"].get(active_channel, []),
            COLORS["roi"],
            3 if kind == "roi" else 1,
        )
        for item in self.profile["bubbles"].get(active_channel, []):
            self._draw_polygon(item["points"], COLORS["bubble"], 3 if kind == "bubble" else 1)
        self._draw_polygon(self.draft, COLORS["draft"], 3)

    def run_result(self) -> None:
        if self._run_process is not None:
            return
        if not self.save():
            return
        complete, missing = profile_completeness(self.profile)
        if not complete:
            messagebox.showwarning("Profile incomplete", "Complete these items first:\n" + "\n".join(missing))
            return
        self.run_button.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set(f"Running profile {self.profile_name}…")

        def worker() -> None:
            command = [sys.executable, str(project_root() / "run_profile.py"), "--profile", self.profile_name]
            try:
                completed = subprocess.run(command, cwd=project_root(), text=True, capture_output=True)
                message = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
                if completed.returncode:
                    error = completed.stderr.strip() or completed.stdout.strip() or f"Exit code {completed.returncode}"
                    self.root.after(0, lambda: self._run_finished(False, error))
                else:
                    self.root.after(0, lambda: self._run_finished(True, message))
            except Exception as exc:
                self.root.after(0, lambda error=str(exc): self._run_finished(False, error))

        threading.Thread(target=worker, daemon=True).start()

    def _run_finished(self, success: bool, detail: str) -> None:
        self._run_process = None
        self.progress.stop()
        self.run_button.configure(state="normal")
        if success:
            runs = sorted((profile_path(self.profile_name).parent / "runs").glob("run_*"))
            result = runs[-1] / "graphs" / "phase_time_series.png" if runs else None
            self.status_var.set(f"Completed: {result or detail}")
            messagebox.showinfo("Result completed", f"Saved to:\n{result or detail}")
        else:
            self.status_var.set("Run failed")
            messagebox.showerror("Run failed", detail[-4000:])

    def close(self) -> None:
        if self.dirty and not messagebox.askyesno("Unsaved changes", "Close without saving changes?"):
            return
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLUID-Space profile editor")
    parser.add_argument("--profile", default="current_baseline")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile, require_complete=args.check)
    if args.check:
        complete, missing = profile_completeness(profile)
        bubble_count = sum(len(profile["bubbles"][channel]) for channel in CHANNELS)
        video_path = resolve_profile_video(args.profile, profile)
        if not video_path.is_file():
            raise FileNotFoundError(f"Profile video not found: {video_path}")
        print(f"Profile: {profile['name']}")
        print(f"Complete: {complete}; missing={missing}")
        print(f"Bubbles: {bubble_count}")
        print(f"Video: {video_path}")
        print(
            f"Frames: {profile['frame']['start']}..{profile['frame']['end']} "
            f"step {profile['frame']['step']}"
        )
        return
    root = tk.Tk()
    FluidSpaceApp(root, args.profile)
    root.mainloop()


if __name__ == "__main__":
    main()
