#!/usr/bin/env python3
"""
End-to-end football pitch homography pipeline.

Inputs:
  - Video file (broadcast/drone style football footage)
  - MOT txt file with rows:
      frame, id, x, y, width, height, confidence, class, visibility, unused

Outputs:
  - AR overlay video (pitch lines projected back to camera view + radar inset)
  - homographies.jsonl
  - player_positions.csv (player feet in image + top-down meters)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO


# FIFA-like standard dimensions (meters)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
CENTER_CIRCLE_RADIUS_M = 9.15


def _penalty_top_y() -> float:
    return (PITCH_WIDTH_M - PENALTY_AREA_WIDTH_M) / 2.0


def _penalty_bottom_y() -> float:
    return (PITCH_WIDTH_M + PENALTY_AREA_WIDTH_M) / 2.0


def _goal_top_y() -> float:
    return (PITCH_WIDTH_M - GOAL_AREA_WIDTH_M) / 2.0


def _goal_bottom_y() -> float:
    return (PITCH_WIDTH_M + GOAL_AREA_WIDTH_M) / 2.0


def _left_arc_rightmost_x() -> float:
    # Penalty spot at x=11m, arc radius=9.15m
    return 11.0 + CENTER_CIRCLE_RADIUS_M


def _right_arc_leftmost_x() -> float:
    return PITCH_LENGTH_M - _left_arc_rightmost_x()


# Must match model's keypoint index definitions.
# Coordinates are in meters in a canonical top-down field coordinate system.
KEYPOINT_TO_WORLD_M: Dict[int, Tuple[float, float]] = {
    0: (0.0, 0.0),  # sideline_top_left
    1: (0.0, _penalty_top_y()),  # big_rect_left_top_pt1
    2: (PENALTY_AREA_DEPTH_M, _penalty_top_y()),  # big_rect_left_top_pt2
    3: (0.0, _penalty_bottom_y()),  # big_rect_left_bottom_pt1
    4: (PENALTY_AREA_DEPTH_M, _penalty_bottom_y()),  # big_rect_left_bottom_pt2
    5: (0.0, _goal_top_y()),  # small_rect_left_top_pt1
    6: (GOAL_AREA_DEPTH_M, _goal_top_y()),  # small_rect_left_top_pt2
    7: (0.0, _goal_bottom_y()),  # small_rect_left_bottom_pt1
    8: (GOAL_AREA_DEPTH_M, _goal_bottom_y()),  # small_rect_left_bottom_pt2
    9: (0.0, PITCH_WIDTH_M),  # sideline_bottom_left
    10: (_left_arc_rightmost_x(), PITCH_WIDTH_M / 2.0),  # left_semicircle_right
    11: (PITCH_LENGTH_M / 2.0, 0.0),  # center_line_top
    12: (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M),  # center_line_bottom
    13: (PITCH_LENGTH_M / 2.0, (PITCH_WIDTH_M / 2.0) - CENTER_CIRCLE_RADIUS_M),  # center_circle_top
    14: (PITCH_LENGTH_M / 2.0, (PITCH_WIDTH_M / 2.0) + CENTER_CIRCLE_RADIUS_M),  # center_circle_bottom
    15: (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0),  # field_center
    16: (PITCH_LENGTH_M, 0.0),  # sideline_top_right
    17: (PITCH_LENGTH_M - PENALTY_AREA_DEPTH_M, _penalty_top_y()),  # big_rect_right_top_pt1
    18: (PITCH_LENGTH_M, _penalty_top_y()),  # big_rect_right_top_pt2
    19: (PITCH_LENGTH_M - PENALTY_AREA_DEPTH_M, _penalty_bottom_y()),  # big_rect_right_bottom_pt1
    20: (PITCH_LENGTH_M, _penalty_bottom_y()),  # big_rect_right_bottom_pt2
    21: (PITCH_LENGTH_M - GOAL_AREA_DEPTH_M, _goal_top_y()),  # small_rect_right_top_pt1
    22: (PITCH_LENGTH_M, _goal_top_y()),  # small_rect_right_top_pt2
    23: (PITCH_LENGTH_M - GOAL_AREA_DEPTH_M, _goal_bottom_y()),  # small_rect_right_bottom_pt1
    24: (PITCH_LENGTH_M, _goal_bottom_y()),  # small_rect_right_bottom_pt2
    25: (PITCH_LENGTH_M, PITCH_WIDTH_M),  # sideline_bottom_right
    26: (_right_arc_leftmost_x(), PITCH_WIDTH_M / 2.0),  # right_semicircle_left
    27: ((PITCH_LENGTH_M / 2.0) - CENTER_CIRCLE_RADIUS_M, PITCH_WIDTH_M / 2.0),  # center_circle_left
    28: ((PITCH_LENGTH_M / 2.0) + CENTER_CIRCLE_RADIUS_M, PITCH_WIDTH_M / 2.0),  # center_circle_right
}


@dataclass
class MotRow:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    cls: int
    vis: float

    @property
    def foot(self) -> Tuple[float, float]:
        return (self.x + (self.w * 0.5), self.y + self.h)

    @property
    def as_xyxy(self) -> Tuple[int, int, int, int]:
        x1 = int(round(self.x))
        y1 = int(round(self.y))
        x2 = int(round(self.x + self.w))
        y2 = int(round(self.y + self.h))
        return x1, y1, x2, y2


def parse_mot(mot_path: Path, min_conf: float = 0.0) -> Dict[int, List[MotRow]]:
    df = pd.read_csv(
        mot_path,
        header=None,
        names=[
            "frame",
            "id",
            "x",
            "y",
            "w",
            "h",
            "confidence",
            "class",
            "visibility",
            "unused",
        ],
    )
    mot_by_frame: Dict[int, List[MotRow]] = {}
    for r in df.itertuples(index=False):
        conf = float(r.confidence)
        if conf < min_conf:
            continue
        row = MotRow(
            frame=int(r.frame),
            track_id=int(r.id),
            x=float(r.x),
            y=float(r.y),
            w=float(r.w),
            h=float(r.h),
            conf=conf,
            cls=int(r._7),  # class
            vis=float(r.visibility),
        )
        mot_by_frame.setdefault(row.frame, []).append(row)
    return mot_by_frame


def download_default_model(model_repo: str, model_filename: str) -> Path:
    model_file = hf_hub_download(repo_id=model_repo, filename=model_filename)
    return Path(model_file)


def build_player_mask(shape: Tuple[int, int], players: Iterable[MotRow], dilation_px: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in players:
        x1, y1, x2, y2 = p.as_xyxy
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    if dilation_px > 0:
        kernel = np.ones((dilation_px, dilation_px), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def apply_player_suppression(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Replace dynamic regions with local green-ish median to reduce keypoint false positives.
    suppressed = frame.copy()
    if not np.any(mask):
        return suppressed
    blur = cv2.medianBlur(frame, 11)
    suppressed[mask > 0] = blur[mask > 0]
    return suppressed


def meter_to_canvas(
    x_m: float,
    y_m: float,
    px_per_m: float,
    margin_px: int,
) -> Tuple[float, float]:
    return (x_m * px_per_m + margin_px, y_m * px_per_m + margin_px)


def make_field_layers(px_per_m: float, margin_px: int) -> Tuple[np.ndarray, np.ndarray]:
    field_w = int(round(PITCH_LENGTH_M * px_per_m + 2 * margin_px))
    field_h = int(round(PITCH_WIDTH_M * px_per_m + 2 * margin_px))

    radar_bg = np.full((field_h, field_w, 3), (35, 120, 35), dtype=np.uint8)
    line_layer = np.zeros((field_h, field_w, 3), dtype=np.uint8)

    def pt(xm: float, ym: float) -> Tuple[int, int]:
        x, y = meter_to_canvas(xm, ym, px_per_m=px_per_m, margin_px=margin_px)
        return int(round(x)), int(round(y))

    line_color = (255, 255, 255)
    line_thickness = max(1, int(round(px_per_m * 0.15)))

    # Outer boundary
    cv2.rectangle(line_layer, pt(0, 0), pt(PITCH_LENGTH_M, PITCH_WIDTH_M), line_color, line_thickness)

    # Halfway line
    cv2.line(line_layer, pt(PITCH_LENGTH_M / 2, 0), pt(PITCH_LENGTH_M / 2, PITCH_WIDTH_M), line_color, line_thickness)

    # Center circle + spot
    center = pt(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2)
    center_r = int(round(CENTER_CIRCLE_RADIUS_M * px_per_m))
    cv2.circle(line_layer, center, center_r, line_color, line_thickness)
    cv2.circle(line_layer, center, max(1, int(round(px_per_m * 0.2))), line_color, thickness=-1)

    # Left penalty + goal boxes
    cv2.rectangle(
        line_layer,
        pt(0, _penalty_top_y()),
        pt(PENALTY_AREA_DEPTH_M, _penalty_bottom_y()),
        line_color,
        line_thickness,
    )
    cv2.rectangle(
        line_layer,
        pt(0, _goal_top_y()),
        pt(GOAL_AREA_DEPTH_M, _goal_bottom_y()),
        line_color,
        line_thickness,
    )
    cv2.circle(line_layer, pt(11.0, PITCH_WIDTH_M / 2), max(1, int(round(px_per_m * 0.2))), line_color, thickness=-1)

    # Right penalty + goal boxes
    cv2.rectangle(
        line_layer,
        pt(PITCH_LENGTH_M - PENALTY_AREA_DEPTH_M, _penalty_top_y()),
        pt(PITCH_LENGTH_M, _penalty_bottom_y()),
        line_color,
        line_thickness,
    )
    cv2.rectangle(
        line_layer,
        pt(PITCH_LENGTH_M - GOAL_AREA_DEPTH_M, _goal_top_y()),
        pt(PITCH_LENGTH_M, _goal_bottom_y()),
        line_color,
        line_thickness,
    )
    cv2.circle(line_layer, pt(PITCH_LENGTH_M - 11.0, PITCH_WIDTH_M / 2), max(1, int(round(px_per_m * 0.2))), line_color, thickness=-1)

    radar_bg = cv2.addWeighted(radar_bg, 1.0, line_layer, 1.0, 0.0)
    return radar_bg, line_layer


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H)
    return pts_h.reshape(-1, 2)


def homography_jump_px(H_prev: np.ndarray, H_new: np.ndarray, image_w: int, image_h: int) -> float:
    probe = np.array(
        [
            [0.0, 0.0],
            [image_w * 0.5, 0.0],
            [image_w - 1.0, 0.0],
            [0.0, image_h * 0.5],
            [image_w * 0.5, image_h * 0.5],
            [image_w - 1.0, image_h * 0.5],
            [0.0, image_h - 1.0],
            [image_w * 0.5, image_h - 1.0],
            [image_w - 1.0, image_h - 1.0],
        ],
        dtype=np.float32,
    )
    p_prev = project_points(H_prev, probe)
    p_new = project_points(H_new, probe)
    return float(np.mean(np.linalg.norm(p_prev - p_new, axis=1)))


def smooth_homography(H_prev: Optional[np.ndarray], H_new: np.ndarray, alpha: float) -> np.ndarray:
    if H_prev is None:
        return H_new
    H = (1.0 - alpha) * H_prev + alpha * H_new
    if abs(H[2, 2]) < 1e-8:
        return H_new
    return H / H[2, 2]


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=track_id * 73 + 19)
    return tuple(int(c) for c in rng.integers(80, 255, size=3))


def extract_best_keypoints(
    model: YOLO,
    frame: np.ndarray,
    frame_masked: np.ndarray,
    kp_conf: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
      (xy, conf) for best detection, shapes: (K,2), (K,)
    """

    candidates: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for candidate_frame in (frame, frame_masked):
        results = model.predict(candidate_frame, verbose=False, conf=0.1)
        if not results:
            continue
        r = results[0]
        if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
            continue
        xy_all = r.keypoints.xy.cpu().numpy()  # (N, K, 2)
        conf_all = r.keypoints.conf
        if conf_all is None:
            conf_np = np.ones(xy_all.shape[:2], dtype=np.float32)
        else:
            conf_np = conf_all.cpu().numpy()
        for i in range(xy_all.shape[0]):
            vis_count = int(np.sum(conf_np[i] >= kp_conf))
            candidates.append((vis_count, xy_all[i], conf_np[i]))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, xy, conf = candidates[0]
    return xy, conf


def estimate_homography_from_keypoints(
    xy: np.ndarray,
    conf: np.ndarray,
    kp_conf: float,
    px_per_m: float,
    margin_px: int,
) -> Tuple[Optional[np.ndarray], int, float]:
    src_pts = []
    dst_pts = []
    used = 0

    for idx, (pt_xy, c) in enumerate(zip(xy, conf)):
        if idx not in KEYPOINT_TO_WORLD_M:
            continue
        if float(c) < kp_conf:
            continue
        used += 1
        src_pts.append([float(pt_xy[0]), float(pt_xy[1])])
        wx, wy = KEYPOINT_TO_WORLD_M[idx]
        dx, dy = meter_to_canvas(wx, wy, px_per_m=px_per_m, margin_px=margin_px)
        dst_pts.append([dx, dy])

    if used < 4:
        return None, used, 0.0

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    H, inlier_mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        return None, used, 0.0
    H = H / H[2, 2]
    inlier_ratio = 0.0
    if inlier_mask is not None and len(inlier_mask) > 0:
        inlier_ratio = float(np.sum(inlier_mask)) / float(len(inlier_mask))
    return H, used, inlier_ratio


def draw_ar_lines(frame: np.ndarray, H_img_to_field: np.ndarray, field_line_layer: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    H_field_to_img = np.linalg.inv(H_img_to_field)
    warped = cv2.warpPerspective(field_line_layer, H_field_to_img, (frame.shape[1], frame.shape[0]))
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0

    out = frame.copy()
    out[mask] = cv2.addWeighted(out[mask], 1.0 - alpha, warped[mask], alpha, 0.0)
    return out


def add_radar_inset(frame: np.ndarray, radar_img: np.ndarray, width_px: int = 360, margin_px: int = 16) -> np.ndarray:
    h, w = frame.shape[:2]
    ratio = width_px / max(1, radar_img.shape[1])
    height_px = int(round(radar_img.shape[0] * ratio))
    resized = cv2.resize(radar_img, (width_px, height_px), interpolation=cv2.INTER_AREA)

    x1 = max(0, w - width_px - margin_px)
    y1 = margin_px
    x2 = min(w, x1 + width_px)
    y2 = min(h, y1 + height_px)
    crop = resized[: y2 - y1, : x2 - x1]

    out = frame.copy()
    cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), thickness=-1)
    out[y1:y2, x1:x2] = crop
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football pitch homography pipeline")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video")
    parser.add_argument("--mot", required=True, type=Path, help="Path to MOT txt")
    parser.add_argument("--output_dir", default=Path("outputs"), type=Path, help="Output directory")
    parser.add_argument("--model_path", default=None, type=Path, help="Local YOLO keypoint model .pt")
    parser.add_argument("--model_repo", default="Adit-jain/Soccana_Keypoint", type=str, help="HF model repo")
    parser.add_argument(
        "--model_filename",
        default="Model/weights/best.pt",
        type=str,
        help="HF model filename inside repo",
    )
    parser.add_argument("--device", default=None, type=str, help='Device for YOLO (e.g. "cpu", "0")')
    parser.add_argument("--kp_conf", default=0.45, type=float, help="Keypoint confidence threshold")
    parser.add_argument("--mot_conf", default=0.0, type=float, help="MOT confidence threshold")
    parser.add_argument("--mot_dilate", default=11, type=int, help="Player mask dilation kernel size")
    parser.add_argument("--px_per_m", default=10.0, type=float, help="Radar scale")
    parser.add_argument("--field_margin_px", default=40, type=int, help="Radar field margin")
    parser.add_argument("--ema_alpha", default=0.35, type=float, help="Homography EMA smoothing")
    parser.add_argument("--max_h_jump_px", default=170.0, type=float, help="Reject larger homography jumps")
    parser.add_argument("--frame_stride", default=1, type=int, help="Process every Nth frame")
    parser.add_argument("--radar_width_px", default=360, type=int, help="Inset radar width in output video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.mot.exists():
        raise FileNotFoundError(f"MOT file not found: {args.mot}")

    model_path = args.model_path
    if model_path is None:
        model_path = download_default_model(args.model_repo, args.model_filename)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[INFO] Loading keypoint model: {model_path}")
    model = YOLO(str(model_path))
    if args.device:
        model.to(args.device)

    print(f"[INFO] Reading MOT file: {args.mot}")
    mot_by_frame = parse_mot(args.mot, min_conf=args.mot_conf)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = args.output_dir / f"{args.video.stem}_ar_radar.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25.0,
        (frame_w, frame_h),
    )

    radar_bg, field_line_layer = make_field_layers(args.px_per_m, args.field_margin_px)
    radar_h, radar_w = radar_bg.shape[:2]

    homography_jsonl = args.output_dir / "homographies.jsonl"
    player_csv = args.output_dir / "player_positions.csv"

    H_prev: Optional[np.ndarray] = None
    player_rows: List[List[object]] = []

    with homography_jsonl.open("w", encoding="utf-8") as hlog:
        for idx in tqdm(range(total_frames), desc="Processing frames"):
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx_1 = idx + 1  # MOT is typically 1-indexed

            if args.frame_stride > 1 and (idx % args.frame_stride != 0):
                if H_prev is not None:
                    drawn = draw_ar_lines(frame, H_prev, field_line_layer)
                    writer.write(add_radar_inset(drawn, radar_bg, width_px=args.radar_width_px))
                else:
                    writer.write(frame)
                continue

            players = mot_by_frame.get(frame_idx_1, [])
            mask = build_player_mask((frame_h, frame_w), players, dilation_px=args.mot_dilate)
            frame_masked = apply_player_suppression(frame, mask)

            detection = extract_best_keypoints(model, frame, frame_masked, kp_conf=args.kp_conf)
            H_curr = None
            used_kpts = 0
            inlier_ratio = 0.0
            if detection is not None:
                xy, conf = detection
                H_curr, used_kpts, inlier_ratio = estimate_homography_from_keypoints(
                    xy=xy,
                    conf=conf,
                    kp_conf=args.kp_conf,
                    px_per_m=args.px_per_m,
                    margin_px=args.field_margin_px,
                )

            if H_curr is not None and H_prev is not None:
                jump = homography_jump_px(H_prev, H_curr, image_w=frame_w, image_h=frame_h)
                if jump > args.max_h_jump_px:
                    H_curr = None

            if H_curr is not None:
                H_est = smooth_homography(H_prev, H_curr, alpha=args.ema_alpha)
                H_prev = H_est

            radar_frame = radar_bg.copy()
            annotated = frame.copy()

            homography_status = "missing"
            if H_prev is not None:
                homography_status = "ok"
                annotated = draw_ar_lines(annotated, H_prev, field_line_layer)

                if players:
                    feet = np.array([p.foot for p in players], dtype=np.float32)
                    feet_on_field = project_points(H_prev, feet)

                    for p, topdown in zip(players, feet_on_field):
                        tx, ty = float(topdown[0]), float(topdown[1])
                        in_bounds = 0.0 <= tx < radar_w and 0.0 <= ty < radar_h
                        if in_bounds:
                            color = id_to_color(p.track_id)
                            cv2.circle(radar_frame, (int(round(tx)), int(round(ty))), 4, color, thickness=-1)
                            cv2.putText(
                                radar_frame,
                                str(p.track_id),
                                (int(round(tx)) + 5, int(round(ty)) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                        x_m = (tx - args.field_margin_px) / args.px_per_m
                        y_m = (ty - args.field_margin_px) / args.px_per_m
                        player_rows.append(
                            [
                                frame_idx_1,
                                p.track_id,
                                p.conf,
                                p.vis,
                                p.foot[0],
                                p.foot[1],
                                x_m,
                                y_m,
                                int(in_bounds),
                            ]
                        )

            annotated = add_radar_inset(annotated, radar_frame, width_px=args.radar_width_px)
            cv2.putText(
                annotated,
                f"Frame: {frame_idx_1} | H: {homography_status} | keypoints: {used_kpts} | inliers: {inlier_ratio:.2f}",
                (18, frame_h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)

            payload = {
                "frame": frame_idx_1,
                "status": homography_status,
                "used_keypoints": used_kpts,
                "inlier_ratio": inlier_ratio,
                "H_img_to_field": H_prev.tolist() if H_prev is not None else None,
            }
            hlog.write(json.dumps(payload) + "\n")

    cap.release()
    writer.release()

    with player_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "frame",
                "id",
                "det_conf",
                "visibility",
                "foot_x_img",
                "foot_y_img",
                "x_m",
                "y_m",
                "inside_field",
            ]
        )
        wr.writerows(player_rows)

    print("[DONE] Output video:", out_video)
    print("[DONE] Homography log:", homography_jsonl)
    print("[DONE] Player positions:", player_csv)


if __name__ == "__main__":
    main()
