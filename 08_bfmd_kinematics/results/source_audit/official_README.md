# BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning

[![Conference](https://img.shields.io/badge/CVSports-2026-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)]()

Official repository for **BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning**.

🌐 **[Dataset Page](https://ning-d.github.io/BFMD-Dataset/)** · 📥 **[Download (Google Drive)](https://drive.google.com/drive/folders/1wQr4DpMbx-e8jFnvOH9WUJXxfXgTC1IJ?usp=drive_link)** · 📄 **[Paper (CVSports @ CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Ding_BFMD_A_Full-Match_Badminton_Dense_Dataset_for_Dense_Shot_Captioning_CVPRW_2026_paper.pdf)**

## Overview

Understanding tactical dynamics in badminton requires analyzing entire matches rather than isolated clips. However, existing badminton datasets mainly focus on short clips or task-specific annotations and rarely provide full-match data with dense multimodal annotations. This limitation makes it difficult to generate accurate shot captions and perform match-level analysis.

To address this limitation, we introduce **BFMD**, the first full-match badminton dense dataset, containing:

- **19 broadcast matches**
- **20+ hours** of play
- **1,687 rallies**
- **16,751 hit events**
- both **singles and doubles**
- a shot caption for each hit event

BFMD provides hierarchical annotations including:

- match segments
- rally events
- shot types
- shuttle trajectories
- player pose keypoints
- shot captions

In addition, we develop a **VideoMAE-based multimodal captioning framework with Semantic Feedback**, which leverages shot semantics to guide caption generation and improve semantic consistency.

Experimental results show that multimodal modeling and semantic feedback improve shot caption quality over RGB-only baselines. We also demonstrate the potential of BFMD for analyzing the temporal evolution of tactical patterns across full matches.

---

## Download

Full dataset documentation is available on the **[Dataset Page](https://ning-d.github.io/BFMD-Dataset/)**.

The BFMD dataset (annotations, match index, and helper scripts) is available on Google Drive:

**[📥 BFMD Dataset (Google Drive)](https://drive.google.com/drive/folders/1wQr4DpMbx-e8jFnvOH9WUJXxfXgTC1IJ?usp=drive_link)**

Videos are not redistributed for copyright reasons — they can be re-downloaded from the official [BWF TV](https://www.youtube.com/c/bwftv) YouTube channel with the provided `download_youtube.py` script (see [Getting the videos](#getting-the-videos)).

---

## Data Package

The released data package contains **12 men's singles matches**
(four BWF World Tour 2025 events: KAPAL API Indonesia Open / PETRONAS Malaysia Open /
VICTOR China Open / YONEX All England Open) + **7 men's doubles matches** (2023–2024),
totaling **1,058 rallies and 11,301 shots**.
All annotations are real JSON files (~1.8 GB); videos can be re-downloaded
from YouTube with `download_youtube.py`.

```
BFMD_data/
├── README.md                    ← this file
├── vis.py                       ← visualization script (render videos/frames with all annotations overlaid)
├── download_youtube.py          ← download source videos from YouTube (BWF TV)
├── Badminton_video_list.csv     ← YouTube links for all 19 matches (each verified by duration)
├── match_index.csv              ← per-match index (event, round, players, rally count, video size)
├── videos/<tournament>/<match_name>.mp4      # 12 singles full-match videos (~9.4 GB)
├── videos_doubles/<match_name>.mp4           # 7 doubles full-match videos
└── annotations/
    ├── metadata/          # match-level info: players, games, rally intervals, score events
    ├── court/             # static court corners + net keypoints (one per match)
    ├── court_perframe/    # per-frame (~1 Hz sampling) court corner detections
    ├── player_bbox/       # full-match top/bottom player box tracks (Label Studio, percent coords)
    ├── pose/              # per-rally player bboxes + 17-keypoint COCO poses
    ├── shuttle/           # TrackNet shuttlecock trajectories (percent coords)
    ├── shot_type/         # timeline labels: rally/replay/hawk-eye intervals + per-shot type (manual)
    ├── hits_doubles/      # doubles hit-event GT (Label Studio export; hit moments, no shot-type labels)
    └── hit_inferred/      # hit event sequences inferred from shot_type (frame/type/player/side)
```

All files are keyed by `<match_name>` (the file name is the match name), e.g.
`KAPAL-API-Indonesia-Open-2025-Anders-Antonsen-DEN-3-vs.-Chou-Tien-Chen-TPE-6-F`.

### Global conventions

| Convention | Value |
|---|---|
| Frame index | **0-based decode order**, based on the full broadcast video |
| Video spec | 1280×720, 30 fps, one mp4 per match |
| Rally numbering | 0-based within each game (`game` ∈ {1,2,3}, `rally` ∈ {0,1,…}) |
| Coordinate systems | Vary by annotation source, see table below ⚠️ |

| Annotation | Coordinate units |
|---|---|
| `court` / `court_perframe` | pixels |
| `bbox` in `pose` | pixels |
| `keypoints` in `pose` | **normalized 0–1** (multiply by width/height for pixels) |
| `player_bbox` / `shuttle` | **percent 0–100** (multiply by W/100, H/100 for pixels) |

### Annotation details

#### 1. Match structure — `metadata/<match>.json`
Top level is a single-element list. Core fields:

```jsonc
{
  "match_name": "...", "player1_name": "Anders Antonsen", "player2_name": "Chou Tien Chen",
  "game1_top_player": "player2",          // player on the far side (top of frame) in game 1
  "games": {
    "game1": {
      "start_label": 16192, "end_label": 68065,
      "rallies": [{"start": 16192, "end": 16830}, ...],   // frame interval of each rally
      "score_events": [{"score_after_rally": {...}, "winner": "player2"}, ...],
      "final_score": {...}, "game_winner": "..."
    }
  },
  "auto_extracted": {"num_rallies": 77, "intervals_11": [...], "intervals_21": [...]},
  "match_score": {"player1": 2, "player2": 0}, "match_winner": "player1"
}
```

#### 2. Court geometry — `court/<match>_court.json`, `court_perframe/<match>.json`
- `court`: one-time calibration of the four court corners
  `court.{top_left,top_right,bottom_left,bottom_right}` and the net
  `net.{left_base,right_base,left_top,right_top}` (pixels). **Valid only for the main camera.**
- `court_perframe`: automatic detections at ~1 frame per second,
  `frames: [{frame_idx, corners: [TL,TR,BR,BL], fit_score, flag}]`. `flag` takes three values:
  - `valid` — main camera, corners reliable (fit_score ≥ 0.5);
  - `refit` — after the broadcast cuts to a secondary camera / zoomed-in shot, corners repaired
    by a global court re-detection pipeline (fit_score is the white-line coverage of the model lines, ≥ 0.65);
  - `no_court` — court not visible or detection unreliable (close-up / replay / far side too dark);
    **corners unusable** — the court layer should be hidden when visualizing
    (`vis.py` already does this).

  Repair pipeline: white-line mask → Hough segments → vanishing points
  clustered into two families → hypothesis search matching detected lines to court-model lines
  to solve a homography → scored by full-model-line coverage → per-line normal ICP refinement,
  with geometric guardrails against degenerate solutions; camera poses already solved for the
  same match are reused.

#### 3. Players — `player_bbox/<match>.json`, `pose/<match>.json`
- `player_bbox` (Label Studio export): two tracks in `annotations[0].result[]`,
  `value.labels` is `Top_Player` / `Bottom_Player`,
  `value.sequence: [{frame, x, y, width, height, enabled}]`, percent coordinates,
  interpolated per frame.
- `pose`: organized by rally, keys like `rally_045_74939_75211` (index_startFrame_endFrame):

```jsonc
{"videos": {"rally_045_74939_75211": {
    "fps": 30.0, "width": 1280, "height": 720,
    "frames": {"74939": [                      // key = full-match frame number; 2 players per frame
        {"bbox": [x1, y1, x2, y2],             // pixels
         "keypoints": [[x, y, score], ...]}    // 17-keypoint COCO, x/y normalized 0–1
    ]}}}}
```

#### 4. Shuttlecock trajectory — `shuttle/<match>.json`
TrackNet predictions, `predictions[0].result[0].value.sequence: [{frame, x, y, width, height, enabled}]`,
percent coordinates; shuttle center = `(x + width/2, y + height/2)`.

#### 5. Hits and shot types — `shot_type/<match>.json`, `hit_inferred/<match>.json`
- `shot_type` (manual timeline annotation, Label Studio): each entry in
  `annotations[0].result[]` is
  `{value: {ranges: [{start, end}], timelinelabels: ["smash"]}}`. Labels include structural
  intervals (`Rally`, `replay`, `hawk_eye_challenge`, `11/21-point interval`) and shot types
  (`serve, flick_serve, clear, drop, smash, drive, net_shot, net_kill, lift, push, press, block, net_hit, hit, bounce`).
- `hit_inferred` (inferred from the above + player positions, **recommended for direct use**):

```jsonc
{"n_hits": 978, "hits": [
  {"frame": 16201, "shot_type": "serve", "side": "top",
   "player": "Chou Tien Chen", "game": 1, "rally": 0}, ...]}
```

#### 6. Doubles — `hits_doubles/<match>.json` + `videos_doubles/`
Manual timeline GT for 7 men's doubles matches (Label Studio export, same format as
`shot_type`). Unlike singles, shots are **not** labeled with specific shot types — labels are
generic hit events (`hit`, `bounce`, `net_hit`) plus structural intervals
(`Rally`, `replay`, `hawk_eye_challenge`, `11/21-point interval`).
No other annotation categories yet.

### Getting the videos

Re-download the source videos using `Badminton_video_list.csv`
(19 matches, all verified against the **BWF TV channel with exact duration match**):

```bash
pip install yt-dlp                      # ffmpeg required for stream merging
python download_youtube.py --list       # show the plan (existing videos marked "have")
python download_youtube.py              # download all missing videos to the paths the annotations expect
python download_youtube.py --match Antonsen   # only download specific matches
```

Downloads are fixed at ≤720p merged mp4, aligned with the 1280×720 / 30 fps decode frame
numbering used by the annotations.
Video copyright belongs to [BWF TV](https://www.youtube.com/c/bwftv); do not redistribute.

### Visualization — `vis.py`

Dependencies: `pip install opencv-python numpy`.

```bash
python vis.py --list                                  # all matches + annotation completeness
python vis.py --list --match Antonsen                 # rally table for one match (frame intervals/shot counts/scores)

# Render an overlay video for one rally (court lines, net, player boxes, skeletons,
# shuttle trail, hit flashes, bottom hit timeline)
python vis.py --match "Indonesia-Open-2025-Anders" --game 1 --rally 2 --out rally.mp4

# Use per-frame court corners instead of the static calibration
python vis.py --match Antonsen --game 1 --rally 2 --perframe-court

# Render any single frame (automatically locates the containing rally and its annotations)
python vis.py --match Antonsen --frame 19250 --out frame.jpg
```

`--match` supports substring matching and lists candidates on ambiguity. Overlay legend:
green quadrilateral = court, yellow line = net, orange box = Top player,
red box = Bottom player, green skeleton = 17-keypoint pose, yellow trail = shuttle trajectory;
at each hit, `shot_type (player)` flashes above the hitter, and the top banner shows
game/rally/score.

---

## Method

We provide a **VideoMAE-based multimodal captioning framework** with a **Semantic Feedback** mechanism.

### Main components

- **Visual encoder** based on VideoMAE
- **Multimodal fusion** of visual and structured cues
- **Transformer-based caption decoder**
- **Semantic Feedback module** for improving semantic consistency

This framework serves as a baseline for dense shot captioning on BFMD.

---

## Repository Structure

You can adjust this part based on your actual project files.

```bash
BFMD/
├── data/                 # dataset files or links / metadata
├── annotations/          # annotation files
├── checkpoints/          # trained model checkpoints
├── scripts/              # preprocessing / training / evaluation scripts
├── src/                  # model source code
├── README.md
├── requirements.txt
└── environment.yml
```

---

## Citation

If you use BFMD in your research, please cite our
[CVPR 2026 Workshop (CVSports) paper](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Ding_BFMD_A_Full-Match_Badminton_Dense_Dataset_for_Dense_Shot_Captioning_CVPRW_2026_paper.pdf):

```bibtex
@InProceedings{Ding_2026_CVPR,
    author    = {Ding, Ning and Fujii, Keisuke and Tamaki, Toru},
    title     = {BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {9909-9918}
}
```