# Football Pitch Homography Pipeline

Colab-friendly pipeline that:

1. Reads a football video (`.mp4`) and MOT tracks (`mot.txt`)
2. Detects pitch keypoints with a pretrained model
3. Estimates frame-wise homography (image -> top-down field)
4. Projects player feet to radar coordinates
5. Draws AR pitch lines back on video

The pipeline is built for input format:

`frame, id, x, y, width, height, confidence, class, visibility, unused`

---

## Quick Run (Google Colab)

```bash
!git clone <YOUR_REPO_URL>
%cd <YOUR_REPO_DIR>
!pip install -r requirements.txt
```

Upload files to Colab (or mount Drive):

- `stal2.mp4`
- `mot.txt`

Run:

```bash
!python3 run_pipeline.py \
  --video stal2.mp4 \
  --mot mot.txt \
  --output_dir outputs
```

Outputs:

- `outputs/stal2_ar_radar.mp4`
- `outputs/homographies.jsonl`
- `outputs/player_positions.csv`

---

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run_pipeline.py --video stal2.mp4 --mot mot.txt --output_dir outputs
```

---

## Notes

- Default keypoint model is downloaded automatically from Hugging Face:
  - Repo: `Adit-jain/Soccana_Keypoint`
  - File: `Model/weights/best.pt`
- If you already have a local model file:

```bash
python3 run_pipeline.py \
  --video stal2.mp4 \
  --mot mot.txt \
  --model_path /path/to/best.pt \
  --output_dir outputs
```

- Useful tuning flags:
  - `--kp_conf 0.45` keypoint confidence threshold
  - `--mot_dilate 11` MOT mask dilation
  - `--ema_alpha 0.35` temporal smoothing
  - `--max_h_jump_px 170` reject abrupt homography jumps

---

## What this pipeline does internally

- Uses MOT boxes to mask dynamic players during pitch calibration
- Runs keypoint detection on original + masked frame
- Solves homography with RANSAC from visible pitch keypoints
- Applies temporal smoothing and jump rejection for stability
- Maps player footpoints (`x + w/2`, `y + h`) to top-down space

This is a robust baseline and can be extended with:
- deep line segmentation
- EKF/smoother over camera pose
- confidence-aware AR rendering
