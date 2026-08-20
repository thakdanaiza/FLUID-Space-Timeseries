# FLUID-Space

FLUID-Space creates channel-level phase-index time series from experiment videos using a fixed CAD layout. You select a video and frame range, review the familiar setup for each channel, and run the analysis.

The application supports Windows 10/11 and current Intel or Apple Silicon Macs. Conda is not required.

## Documentation

- Open `docs/FLUID-Space_User_Guide_and_Methods.docx` for the complete illustrated-style user guide, quality-control checklist, analysis explanation, and manuscript-ready Methods template.
- Open `METHODS.md` for a plain-text research methods reference that can be copied and adapted for individual studies.

The Methods template contains placeholders in square brackets. Replace them with the actual video, frame, software, acquisition, and study details before using the text in a manuscript.

## Citing FLUID-Space

Cite the exact FLUID-Space release used for the analysis. Do not cite only this README or an unversioned working branch.

Example Methods sentence:

> Image analysis was performed using FLUID-Space v1.0.0 (Author et al., 2026), a CAD-authoritative image-analysis workflow. The registered CAD geometry was intersected with channel-specific regions of interest and Interest Zones before calibration regions and manually annotated gas bubbles were excluded.

Example software reference before a DOI is available:

> Author Surname, Initials., Author Surname, Initials., & Author Surname, Initials. (2026). *FLUID-Space* (Version 1.0.0) [Computer software]. Laboratory or institution. Repository URL

Preferred reference after archiving the release on Zenodo or another DOI-issuing repository:

> Author Surname, Initials., Author Surname, Initials., & Author Surname, Initials. (2026). *FLUID-Space* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.xxxx/zenodo.xxxxxxx

Replace the author names, year, version, institution, repository URL, and DOI with the real release metadata. The citation format may be adjusted to the target journal's reference style, but the software version and persistent release identifier should remain included.

For the first paper that introduces or validates this workflow, describe the algorithm in sufficient detail and cite both the archived software release and relevant underlying methods such as GrabCut/OpenCV. Later papers should cite the validation paper together with the exact FLUID-Space software release used for analysis.

## Before you begin

You need:

- Python 3.12 (64-bit recommended)
- Internet access during the first installation
- Approximately 1 GB of free disk space
- A source video with a resolution of **1090 × 340 pixels**

Download Python from the official website:

- Windows: <https://www.python.org/downloads/windows/>
- macOS: <https://www.python.org/downloads/macos/>

## Install on Windows

1. Install Python 3.12. If the installer offers an option to add Python to `PATH`, enable it.
2. Extract the FLUID-Space ZIP to a normal writable folder, such as Documents.
3. Open the extracted folder.
4. Double-click `setup_env.bat`.
5. Wait until `Environment check passed` is displayed.

Installation is required only once. To open FLUID-Space later, double-click `start_ui.bat`.

## Install on macOS

1. Install Python 3.12 using the official Python.org macOS installer.
2. Extract the FLUID-Space ZIP.
3. Open Terminal.
4. Type `cd `, drag the extracted FLUID-Space folder into Terminal, and press Return.
5. Run:

```bash
bash setup_env.sh
bash start_ui.sh
```

Installation is required only once. To open FLUID-Space later, enter the project folder in Terminal and run:

```bash
bash start_ui.sh
```

You can also enable the Finder launcher once:

```bash
chmod +x start_ui_mac.command
```

After that, open `start_ui_mac.command` from Finder. If macOS shows a security prompt because the file was downloaded, right-click the file, select **Open**, and confirm.

## Use FLUID-Space

### 1. Select a profile

Choose a profile from the **Profile** list at the top of the window.

To create a separate setup without changing the approved baseline, click **Duplicate as…**, enter a new profile name, and use the new profile.

Each profile stores its own:

- Source video
- Start frame, end frame, and frame step
- Selected result channels
- Interest Zones
- Channel ROIs
- Bubble exclusions
- Analysis results

### 2. Select a source video

1. Click **Select video…**.
2. Choose the experiment video.
3. Wait for the video information to appear in the left panel.

FLUID-Space copies the selected video into the active profile. The video remains connected to that profile when the application is reopened.

The video must be **1090 × 340 pixels** because the CAD layout is calibrated to this exact image size.

### 3. Select the analysis frame range

1. Enter **Start**, **End**, and **Step** frame values.
2. Click **Load Start** or **Load End** to inspect either boundary frame on the existing drawing canvas.
3. Confirm that the fixed geometry is suitable at both ends of the selected range.

Frames are sampled as `Start, Start + Step, ...` up to End. End is included only when it falls exactly on the sampling interval.

### 4. Review each channel

Select one channel at a time:

- CH1-1
- CH1-2
- CH2-1
- CH2-2

Only the selected channel's setup is displayed. Use the **CAD overlay** checkbox to show or hide the CAD outline.

Use the **Result channels** checkboxes to choose which channels are processed and exported. At least one result channel must remain selected; the channel radio buttons above them still control only which geometry is being edited.

Use the **Video outputs** checkboxes to optionally create animated Count and/or Percent histogram videos. Both are off by default. Leaving both unchecked keeps the image/CSV/JSON workflow and does not run the video encoder.

### 5. Edit the analysis setup

Select the item you want to edit:

- **Interest Zone** — defines the useful area independently for the selected channel
- **Channel ROI** — limits the area used for the selected channel
- **Bubble** — retained for the familiar annotation workflow, but not applied to time-series results

Drawing controls:

- Left click: add a point
- Right click or Enter: close the polygon
- Mouse wheel: zoom
- Middle-button drag: pan
- Esc: cancel the current unfinished polygon
- Ctrl+Z: remove the latest unfinished point
- Ctrl+S: save the profile

Use **Clear current** carefully. For Interest Zone or Channel ROI, it clears the selected polygon. For Bubble, it removes the latest bubble in the selected channel.

The final analysis area is calculated as:

```text
CAD interior ∩ Channel ROI ∩ Channel Interest Zone − Calibration ROI
```

CAD boundaries and internal CAD islands are fixed. Areas outside the CAD interior are never included in the result.

### 6. Save and run

Click **Save** after editing the profile.

Click **Run result** to process the selected frame range. Keep FLUID-Space open until the completion message appears.

## Find the results

Results are saved inside the active profile:

```text
profiles/<profile-name>/runs/run_xxx/
```

The main result image is:

```text
graphs/phase_time_series.png
```

The run folder contains the time-series graph, `time_series.csv`, `phase_histogram.csv`, histogram heatmaps, and `summary.json`. Each time-series subplot shows the channel mean with a shaded `Mean ± 1 SD` band. For every selected channel, the `histograms/` folder contains separate pixel-count and within-frame pixel-percentage heatmaps with elapsed time on the X axis and phase index 1–2 on the Y axis.

If a Video output is selected, the `videos/` folder contains the corresponding `phase_histogram_count_<channel>.mp4` and/or `phase_histogram_percent_<channel>.mp4`. The upper panel is a tight crop of the channel's actual analysis mask, with excluded pixels blacked out. The lower panel is the complete histogram heatmap, and a red line follows the sampled frame being shown. Videos contain no audio and play at `source FPS / Step`, so skipped frames retain the original elapsed-time spacing. This version does not export phase maps, masks, arrays, PDFs, or quality-control images.

## Troubleshooting

### Python was not found

Install Python 3.12 from Python.org, close all Command Prompt or Terminal windows, reopen them, and run the setup again.

### The environment is missing or damaged

Windows: run:

```text
setup_env.bat --recreate
```

macOS: run:

```bash
bash setup_env.sh --recreate
```

This recreates only the application's Python environment. It does not remove profiles or results.

### A package installation fails

Check the internet connection and run the setup script again. It is safe to rerun.

### Tkinter is missing on macOS

Install Python using the official Python.org macOS installer, then recreate the environment:

```bash
bash setup_env.sh --recreate
```

### The video cannot be selected

Confirm that the file is a supported video and its resolution is exactly **1090 × 340 pixels**. Common formats such as MP4, MOV, AVI, MKV, and M4V are available in the file selector.

### The selected frame cannot be loaded

Enter a frame number between `0` and the final frame shown for the selected video.
