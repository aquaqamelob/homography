#!/usr/bin/env python3
"""
Human-in-the-loop football pitch homography pipeline.

This script intentionally avoids YOLO keypoint detection and relies on:
1) Sparse manual keyframe calibration (human picks known field landmarks)
2) Automatic inter-frame homography propagation with ECC
3) Drift checks from line-mask agreement
4) MOT-assisted dynamic masking

Usage:
  # Step 1: create anchors manually
  python3 run_pipeline.py annotate --video stal2.mp4 --frames 1,250,500 --anchors anchors.json

  # Step 2: run pipeline with MOT + anchors
  python3 run_pipeline.py run --video stal2.mp4 --mot mot.txt --anchors anchors.json --output_dir outputs
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


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


LANDMARKS_M: Dict[str, Tuple[float, float]] = {
    # Outer corners / boundaries
    "corner_top_left": (0.0, 0.0),
    "corner_top_right": (PITCH_LENGTH_M, 0.0),
    "corner_bottom_left": (0.0, PITCH_WIDTH_M),
    "corner_bottom_right": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    "halfway_top": (PITCH_LENGTH_M / 2.0, 0.0),
    "halfway_bottom": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M),
    # Left penalty area
    "left_penalty_top_outer": (0.0, _penalty_top_y()),
    "left_penalty_top_inner": (PENALTY_AREA_DEPTH_M, _penalty_top_y()),
    "left_penalty_bottom_outer": (0.0, _penalty_bottom_y()),
    "left_penalty_bottom_inner": (PENALTY_AREA_DEPTH_M, _penalty_bottom_y()),
    # Right penalty area
    "right_penalty_top_inner": (PITCH_LENGTH_M - PENALTY_AREA_DEPTH_M, _penalty_top_y()),
    "right_penalty_top_outer": (PITCH_LENGTH_M, _penalty_top_y()),
    "right_penalty_bottom_inner": (PITCH_LENGTH_M - PENALTY_AREA_DEPTH_M, _penalty_bottom_y()),
    "right_penalty_bottom_outer": (PITCH_LENGTH_M, _penalty_bottom_y()),
    # Goal areas
    "left_goal_top_inner": (GOAL_AREA_DEPTH_M, _goal_top_y()),
    "left_goal_bottom_inner": (GOAL_AREA_DEPTH_M, _goal_bottom_y()),
    "right_goal_top_inner": (PITCH_LENGTH_M - GOAL_AREA_DEPTH_M, _goal_top_y()),
    "right_goal_bottom_inner": (PITCH_LENGTH_M - GOAL_AREA_DEPTH_M, _goal_bottom_y()),
    # Center
    "center_spot": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0),
    "center_circle_top": (PITCH_LENGTH_M / 2.0, (PITCH_WIDTH_M / 2.0) - CENTER_CIRCLE_RADIUS_M),
    "center_circle_bottom": (PITCH_LENGTH_M / 2.0, (PITCH_WIDTH_M / 2.0) + CENTER_CIRCLE_RADIUS_M),
    "center_circle_left": ((PITCH_LENGTH_M / 2.0) - CENTER_CIRCLE_RADIUS_M, PITCH_WIDTH_M / 2.0),
    "center_circle_right": ((PITCH_LENGTH_M / 2.0) + CENTER_CIRCLE_RADIUS_M, PITCH_WIDTH_M / 2.0),
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
    suppressed = frame.copy()
    if np.any(mask):
        blur = cv2.medianBlur(frame, 11)
        suppressed[mask > 0] = blur[mask > 0]
    return suppressed


def meter_to_canvas(x_m: float, y_m: float, px_per_m: float, margin_px: int) -> Tuple[float, float]:
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

    cv2.rectangle(line_layer, pt(0, 0), pt(PITCH_LENGTH_M, PITCH_WIDTH_M), line_color, line_thickness)
    cv2.line(line_layer, pt(PITCH_LENGTH_M / 2, 0), pt(PITCH_LENGTH_M / 2, PITCH_WIDTH_M), line_color, line_thickness)

    center = pt(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2)
    center_r = int(round(CENTER_CIRCLE_RADIUS_M * px_per_m))
    cv2.circle(line_layer, center, center_r, line_color, line_thickness)
    cv2.circle(line_layer, center, max(1, int(round(px_per_m * 0.2))), line_color, thickness=-1)

    cv2.rectangle(
        line_layer,
        pt(0, _penalty_top_y()),
        pt(PENALTY_AREA_DEPTH_M, _penalty_bottom_y()),
        line_color,
        line_thickness,
    )
    cv2.rectangle(
        line_layer,
        pt(PITCH_LENGTH_M - PENALTY_AREA_DEPTH_M, _penalty_top_y()),
        pt(PITCH_LENGTH_M, _penalty_bottom_y()),
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
    cv2.rectangle(
        line_layer,
        pt(PITCH_LENGTH_M - GOAL_AREA_DEPTH_M, _goal_top_y()),
        pt(PITCH_LENGTH_M, _goal_bottom_y()),
        line_color,
        line_thickness,
    )
    cv2.circle(line_layer, pt(11.0, PITCH_WIDTH_M / 2), max(1, int(round(px_per_m * 0.2))), line_color, thickness=-1)
    cv2.circle(
        line_layer,
        pt(PITCH_LENGTH_M - 11.0, PITCH_WIDTH_M / 2),
        max(1, int(round(px_per_m * 0.2))),
        line_color,
        thickness=-1,
    )
    radar_bg = cv2.addWeighted(radar_bg, 1.0, line_layer, 1.0, 0.0)
    return radar_bg, line_layer


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H)
    return pts_h.reshape(-1, 2)


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=track_id * 73 + 19)
    return tuple(int(c) for c in rng.integers(80, 255, size=3))


def parse_frames_arg(frames_text: str) -> List[int]:
    parts = [p.strip() for p in frames_text.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    out = sorted(set(out))
    return out


def build_homography_from_points(
    image_points: Sequence[Sequence[float]],
    world_points_m: Sequence[Sequence[float]],
    px_per_m: float,
    margin_px: int,
) -> Optional[np.ndarray]:
    if len(image_points) < 4 or len(world_points_m) < 4:
        return None
    src = np.asarray(image_points, dtype=np.float32)
    dst = np.asarray([meter_to_canvas(p[0], p[1], px_per_m, margin_px) for p in world_points_m], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        return None
    return H / H[2, 2]


def load_anchor_homographies(anchors_path: Path, px_per_m: float, margin_px: int) -> Dict[int, np.ndarray]:
    payload = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors = payload.get("anchors", [])
    out: Dict[int, np.ndarray] = {}
    for item in anchors:
        frame_idx = int(item["frame"])
        points = item.get("points", [])
        image_pts = []
        world_pts = []
        for p in points:
            image = p.get("image")
            if image is None or len(image) != 2:
                continue
            if "world_m" in p and p["world_m"] is not None:
                world = p["world_m"]
            else:
                name = p.get("name")
                world = LANDMARKS_M.get(name)
            if world is None:
                continue
            image_pts.append([float(image[0]), float(image[1])])
            world_pts.append([float(world[0]), float(world[1])])
        H = build_homography_from_points(image_pts, world_pts, px_per_m=px_per_m, margin_px=margin_px)
        if H is not None:
            out[frame_idx] = H
    return out


def compute_line_mask(frame_bgr: np.ndarray, static_mask: Optional[np.ndarray]) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (179, 75, 255))
    # Remove large green regions, keep white markings.
    green = cv2.inRange(hsv, (25, 35, 30), (95, 255, 255))
    line_mask = cv2.bitwise_and(white, cv2.bitwise_not(green))

    if static_mask is not None:
        line_mask = cv2.bitwise_and(line_mask, static_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return line_mask


def estimate_prev_to_curr_warp(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    static_mask_curr: Optional[np.ndarray],
    ecc_iters: int,
) -> Tuple[Optional[np.ndarray], float]:
    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(
            templateImage=prev_gray,
            inputImage=curr_gray,
            warpMatrix=warp,
            motionType=cv2.MOTION_HOMOGRAPHY,
            criteria=criteria,
            inputMask=static_mask_curr,
            gaussFiltSize=5,
        )
        return warp.astype(np.float64), float(cc)
    except cv2.error:
        return None, -1.0


def smooth_homography(H_prev: Optional[np.ndarray], H_new: np.ndarray, alpha: float) -> np.ndarray:
    if H_prev is None:
        return H_new
    H = (1.0 - alpha) * H_prev + alpha * H_new
    if abs(H[2, 2]) < 1e-8:
        return H_new
    return H / H[2, 2]


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


def landmark_help_text() -> str:
    lines = ["Available landmark names:"]
    for name, (x, y) in LANDMARKS_M.items():
        lines.append(f"  - {name:28s} -> ({x:.2f}, {y:.2f})")
    lines.append("At least 4 points per anchor frame are required (6+ recommended).")
    return "\n".join(lines)


def read_frame(video_path: Path, frame_idx_1: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx_1 - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx_1} from {video_path}")
    return frame


def collect_anchor_points_for_frame(frame_bgr: np.ndarray, frame_idx_1: int) -> List[dict]:
    print("\n" + "=" * 80)
    print(f"[ANNOTATE] Frame {frame_idx_1}")
    print(landmark_help_text())
    print('Type a landmark name and click once. Type "done" when finished, "list" to print landmarks again.')

    points: List[dict] = []
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    while True:
        name = input("landmark name> ").strip()
        if name.lower() == "done":
            break
        if name.lower() == "list":
            print(landmark_help_text())
            continue
        if name not in LANDMARKS_M:
            print("[WARN] Unknown landmark name.")
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {frame_idx_1} | Click: {name}")
        ax.axis("off")
        clicked = plt.ginput(1, timeout=0)
        plt.close(fig)

        if not clicked:
            print("[WARN] No click captured.")
            continue
        x, y = clicked[0]
        points.append({"name": name, "image": [float(x), float(y)]})
        print(f"[OK] {name} -> image({x:.1f}, {y:.1f}) world{LANDMARKS_M[name]}")

    return points


def run_annotation_mode(args: argparse.Namespace) -> None:
    frame_ids = parse_frames_arg(args.frames)
    if not frame_ids:
        raise ValueError("No frames provided. Use --frames, e.g. 1,250,500")

    anchors = []
    for frame_idx_1 in frame_ids:
        frame = read_frame(args.video, frame_idx_1)
        points = collect_anchor_points_for_frame(frame, frame_idx_1)
        if len(points) < 4:
            print(f"[WARN] Frame {frame_idx_1} has only {len(points)} points (will be ignored at run time).")
        anchors.append({"frame": frame_idx_1, "points": points})

    payload = {
        "video": str(args.video),
        "anchors": anchors,
        "landmarks_m": LANDMARKS_M,
    }
    args.anchors.parent.mkdir(parents=True, exist_ok=True)
    args.anchors.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote anchors: {args.anchors}")


def line_overlap_score(
    H_img_to_field: np.ndarray,
    field_line_layer: np.ndarray,
    observed_line_mask: np.ndarray,
) -> float:
    H_field_to_img = np.linalg.inv(H_img_to_field)
    warped = cv2.warpPerspective(field_line_layer, H_field_to_img, (observed_line_mask.shape[1], observed_line_mask.shape[0]))
    pred = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
    obs = observed_line_mask > 0
    pred_count = int(np.sum(pred))
    if pred_count == 0:
        return 0.0
    overlap = int(np.sum(pred & obs))
    return float(overlap) / float(pred_count)


def run_pipeline_mode(args: argparse.Namespace) -> None:
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.mot.exists():
        raise FileNotFoundError(f"MOT file not found: {args.mot}")
    if not args.anchors.exists():
        raise FileNotFoundError(f"Anchors file not found: {args.anchors}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading MOT file: {args.mot}")
    mot_by_frame = parse_mot(args.mot, min_conf=args.mot_conf)
    anchor_H = load_anchor_homographies(args.anchors, px_per_m=args.px_per_m, margin_px=args.field_margin_px)
    if not anchor_H:
        raise RuntimeError("No valid anchors found (need 4+ valid points per anchor frame).")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = args.output_dir / f"{args.video.stem}_ar_radar_hitl.mp4"
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
    prev_gray: Optional[np.ndarray] = None
    player_rows: List[List[object]] = []

    with homography_jsonl.open("w", encoding="utf-8") as hlog:
        for idx in tqdm(range(total_frames), desc="Processing frames"):
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx_1 = idx + 1

            players = mot_by_frame.get(frame_idx_1, [])
            player_mask = build_player_mask((frame_h, frame_w), players, dilation_px=args.mot_dilate)
            static_mask = cv2.bitwise_not(player_mask)

            suppressed = apply_player_suppression(frame, player_mask)
            curr_gray = cv2.cvtColor(suppressed, cv2.COLOR_BGR2GRAY)

            H_curr = None
            status = "missing"
            ecc_cc = -1.0
            drift_score = 0.0

            if frame_idx_1 in anchor_H:
                H_curr = anchor_H[frame_idx_1]
                status = "anchor"
            elif H_prev is not None and prev_gray is not None:
                W_prev_to_curr, ecc_cc = estimate_prev_to_curr_warp(
                    prev_gray=prev_gray,
                    curr_gray=curr_gray,
                    static_mask_curr=static_mask,
                    ecc_iters=args.ecc_iters,
                )
                if W_prev_to_curr is not None:
                    try:
                        W_inv = np.linalg.inv(W_prev_to_curr)
                        H_prop = H_prev @ W_inv
                        H_curr = smooth_homography(H_prev, H_prop / H_prop[2, 2], alpha=args.ema_alpha)
                        status = "propagated"
                    except np.linalg.LinAlgError:
                        H_curr = H_prev
                        status = "carry"
                else:
                    H_curr = H_prev
                    status = "carry"

            if H_curr is not None:
                line_mask = compute_line_mask(frame, static_mask=static_mask)
                drift_score = line_overlap_score(H_curr, field_line_layer, line_mask)
                if drift_score < args.min_line_overlap and status != "anchor":
                    # Soft fail: keep previous H but mark low-confidence to request more anchors.
                    if H_prev is not None:
                        H_curr = H_prev
                    status = "low_conf"

            if H_curr is not None:
                H_prev = H_curr

            radar_frame = radar_bg.copy()
            annotated = frame.copy()

            if H_prev is not None:
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
                f"Frame:{frame_idx_1} | H:{status} | ECC:{ecc_cc:.3f} | line:{drift_score:.2f}",
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
                "status": status,
                "ecc_cc": ecc_cc,
                "line_overlap": drift_score,
                "is_anchor_frame": frame_idx_1 in anchor_H,
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
    print("[INFO] If many frames show status=low_conf, add more anchor frames and rerun.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human-in-the-loop football homography pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_annotate = sub.add_parser("annotate", help="Interactive anchor annotation")
    p_annotate.add_argument("--video", required=True, type=Path, help="Path to input video")
    p_annotate.add_argument("--frames", required=True, type=str, help="Comma-separated frame indices (1-based)")
    p_annotate.add_argument("--anchors", required=True, type=Path, help="Output anchors JSON path")

    p_run = sub.add_parser("run", help="Run pipeline with MOT + anchors")
    p_run.add_argument("--video", required=True, type=Path, help="Path to input video")
    p_run.add_argument("--mot", required=True, type=Path, help="Path to MOT txt")
    p_run.add_argument("--anchors", required=True, type=Path, help="Path to anchors JSON")
    p_run.add_argument("--output_dir", default=Path("outputs"), type=Path, help="Output directory")
    p_run.add_argument("--mot_conf", default=0.0, type=float, help="MOT confidence threshold")
    p_run.add_argument("--mot_dilate", default=11, type=int, help="Player mask dilation kernel size")
    p_run.add_argument("--px_per_m", default=10.0, type=float, help="Radar scale")
    p_run.add_argument("--field_margin_px", default=40, type=int, help="Radar field margin")
    p_run.add_argument("--ema_alpha", default=0.35, type=float, help="EMA smoothing of propagated H")
    p_run.add_argument("--ecc_iters", default=60, type=int, help="Max ECC iterations")
    p_run.add_argument("--min_line_overlap", default=0.08, type=float, help="Low-confidence threshold")
    p_run.add_argument("--radar_width_px", default=360, type=int, help="Inset radar width in output video")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "annotate":
        run_annotation_mode(args)
        return
    if args.command == "run":
        run_pipeline_mode(args)
        return
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
