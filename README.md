# Human-in-the-Loop Football Homography Pipeline

Colab-friendly pipeline that avoids YOLO pitch keypoint detection.

It uses:
1. Sparse manual anchor keyframes (human picks visible field landmarks)
2. Automatic homography propagation between frames (ECC + MOT masking)
3. Drift checks using line-mask overlap
4. Player foot projection (`x + w/2`, `y + h`) to top-down radar

Input MOT format:

`frame, id, x, y, width, height, confidence, class, visibility, unused`

---

## Quick Run (Google Colab)

```bash
!git clone <YOUR_REPO_URL>
%cd <YOUR_REPO_DIR>
!pip install -r requirements.txt
```

Upload:
- `stal2.mp4`
- `mot.txt`

### Step 1: Annotate sparse keyframes

Pick 3-8 anchors across camera motion changes (example frames below):

In Colab, run this in a Python cell before annotation so clicks are interactive:

```python
%matplotlib widget
```

```bash
!python3 run_pipeline.py annotate \
  --video stal2.mp4 \
  --frames 1,180,360,540 \
  --anchors anchors.json
```

For each anchor frame:
- type a landmark name (examples: `center_spot`, `halfway_top`, `left_penalty_top_inner`)
- click the corresponding pixel on the frame
- type `done` when finished (4+ points required, 6+ recommended)

### Step 2: Run homography + projection

```bash
!python3 run_pipeline.py run \
  --video stal2.mp4 \
  --mot mot.txt \
  --anchors anchors.json \
  --output_dir outputs
```

Outputs:
- `outputs/stal2_ar_radar_hitl.mp4`
- `outputs/homographies.jsonl`
- `outputs/player_positions.csv`

---

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 run_pipeline.py annotate --video stal2.mp4 --frames 1,180,360 --anchors anchors.json
python3 run_pipeline.py run --video stal2.mp4 --mot mot.txt --anchors anchors.json --output_dir outputs
```

---

## Useful tuning flags (run command)

- `--mot_dilate 11` player-mask dilation
- `--ema_alpha 0.35` temporal smoothing
- `--ecc_iters 60` ECC optimization iterations
- `--min_line_overlap 0.08` drift threshold (raise to be stricter)
- `--px_per_m 10.0` radar map scale

If many frames show `status=low_conf` in `homographies.jsonl`, add more anchor frames and rerun.

---

## Why this is more robust than pure detector-based calibration

- avoids detector hallucinations on grass/repetitive lines
- uses explicit field geometry from human anchors
- separates ego-motion from player motion with MOT masking
- supports partial landmarks (no need for all four pitch corners)
