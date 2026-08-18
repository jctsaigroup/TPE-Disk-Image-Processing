"""
Tracking pipeline: disk detection → trajectory linking → orientation → G² analysis.

Requires: stardist_tf2.10 environment
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np
import pandas as pd
import trackpy as tp
from skimage import measure

from config import Config, configure_logging

logger = logging.getLogger(__name__)


def load_calibration(calibration_file: str) -> np.ndarray:
    """Load the perspective-correction homography matrix from disk."""
    H = np.load(calibration_file)
    if H.shape != (3, 3):
        raise ValueError(f"Expected 3x3 homography matrix, got shape {H.shape}")
    return H


def camera_align(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply perspective correction using a calibrated homography matrix."""
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, H, (width, height))


def load_stardist_model(model_name: str):
    """Load StarDist2D model."""
    from stardist.models import StarDist2D
    return StarDist2D(None, name=model_name)


def load_tracking_result(cfg: Config) -> pd.DataFrame:
    """Load pre-computed tracking result from pickle."""
    pkl_path = os.path.join(cfg.pkl_dir, f"{cfg.exp_folder}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Tracking pickle not found: {pkl_path}")
    
    F_linked = pd.read_pickle(pkl_path)
    logger.info("Loaded %d trajectory rows from %s", len(F_linked), pkl_path)
    return F_linked


def log_sample(n_total: int, n_samples: int, start: int = 1) -> np.ndarray:
    """Log-spaced frame indices in [start, n_total], length == n_samples."""
    n_request = n_samples
    while True:
        log_frames = np.logspace(np.log10(start), np.log10(n_total), n_request)
        frames = np.unique(np.round(log_frames).astype(int))
        if len(frames) >= n_samples:
            return frames[:n_samples] if len(frames) > n_samples else frames
        n_request += n_samples - len(frames) + 5


def build_frame_plan(cfg: Config, max_frame_green: int) -> tuple[list[int], dict[int, int]]:
    """Build list of frames to process and sampling map."""
    if not cfg.tracking.sampling.sample_log_frames:
        all_frames = list(range(1, max_frame_green + 1))
    else:
        s = cfg.tracking.sampling
        local_sample = log_sample(s.frames_per_trial, s.n_sample_per_trial)
        local_sample[-1] = s.frames_per_trial
        
        all_frames = []
        for trial_idx in range(s.n_trials):
            for local_frame in local_sample:
                global_frame = trial_idx * s.frames_per_trial + int(local_frame)
                all_frames.append(global_frame)
    
    # Apply frame selection mode
    mode = cfg.tracking.frame_selection.get("mode", "all")
    value = cfg.tracking.frame_selection.get("value")
    
    if mode == "random" and value:
        import random
        frames_to_process = sorted(random.sample(all_frames, min(int(value), len(all_frames))))
    elif mode == "first" and value:
        frames_to_process = all_frames[:int(value)]
    elif mode == "single" and value:
        frames_to_process = [int(value)] if int(value) in all_frames else []
    else:
        frames_to_process = all_frames

    if not cfg.tracking.sampling.sample_log_frames:
        return frames_to_process, {}

    s = cfg.tracking.sampling
    local_sample = log_sample(s.frames_per_trial, s.n_sample_per_trial)
    local_sample[-1] = s.frames_per_trial

    frame_map = []
    sample_idx = 1
    for trial_idx in range(s.n_trials):
        for local_frame in local_sample:
            global_frame = trial_idx * s.frames_per_trial + int(local_frame)
            if global_frame in frames_to_process:
                frame_map.append((sample_idx, global_frame, trial_idx, int(local_frame)))
            sample_idx += 1

    frame_map_df = pd.DataFrame(
        frame_map, columns=["sample_idx", "global_frame", "trial", "local_frame"]
    )
    global_to_sample = dict(zip(frame_map_df["global_frame"], frame_map_df["sample_idx"]))
    
    return frames_to_process, global_to_sample


def detect_disks(
    cfg: Config,
    model,
    H: np.ndarray,
    frames_to_process: list[int],
    global_to_sample: dict[int, int],
    src_helpers,
) -> pd.DataFrame:
    """Run StarDist over frames, filter by bounds/eccentricity, apply DoG refinement."""
    roi = cfg.roi_as_tuple()
    d = cfg.detection
    axis_norm = (0, 1)

    kernels = {
        d.large_radius_px: src_helpers.dog_kernel(49, delta=3, sigma=2),
        d.small_radius_px: src_helpers.dog_kernel(40, delta=3, sigma=2),
    }

    from csbdeep.utils import normalize

    records = []
    n_frames = len(frames_to_process)
    for frame_idx, frame in enumerate(frames_to_process, 1):
        print(f"\r  Detecting disks: frame {frame_idx}/{n_frames}...", end='', flush=True)

        Ig = I[roi[0]:roi[1], roi[2]:roi[3]]
        input_image = normalize(Ig, 1, 99.8, axis=axis_norm)

        mask, detail = model.predict_instances(
            input_image, n_tiles=model._guess_n_tiles(input_image), show_tile_progress=False
        )
        props = measure.regionprops(mask)

        frame_detections = []
        for region in props:
            y, x = region.centroid
            area = region.area
            ecc = region.eccentricity

            x_lo, x_hi = d.x_bounds
            if not (x_lo < x < x_hi and y > d.y_min and ecc < d.eccentricity_max):
                continue

            rpx = d.large_radius_px if area >= d.area_threshold else d.small_radius_px

            frame_detections.append({
                "log_sampled_frames": global_to_sample.get(frame) if cfg.tracking.sampling.sample_log_frames else None,
                "frame": frame,
                "x": x,
                "y": y,
                "area": area,
                "ecc": ecc,
                "rpx": rpx,
            })

        if frame_detections:
            temp_df = pd.DataFrame(frame_detections)
            refined_list = src_helpers.refine_frame_centers(temp_df, I, kernels, roi)
            records.extend([s.to_dict() for s in refined_list])

    print()  # Newline after progress
    if not records:
        raise RuntimeError("No detections produced across any processed frame")

    return pd.DataFrame(records)


def link_trajectories(cfg: Config, df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Split into per-trial segments, link each with Trackpy, recombine with offset IDs."""
    s = cfg.tracking.sampling
    lk = cfg.linking

    if s.sample_log_frames:
        n_segments = s.n_trials
        frames_per_seg = s.n_sample_per_trial
    else:
        n_segments = s.n_trials
        frames_per_seg = s.frames_per_trial

    df_filtered = df_filtered.copy()
    df_filtered["_segment"] = (df_filtered["frame"] - 1) // frames_per_seg

    linked_segments = []
    for seg_idx in range(n_segments):
        df_seg = df_filtered[df_filtered["_segment"] == seg_idx].drop(columns="_segment").copy()
        if df_seg.empty:
            logger.warning("Segment %d has no detections — skipping", seg_idx)
            continue

        linked_seg = tp.link(df_seg, search_range=lk.search_range, memory=lk.memory)
        linked_seg["particle"] = linked_seg["particle"] + seg_idx * lk.id_offset

        if s.sample_log_frames:
            linked_seg["trial"] = seg_idx

        linked_segments.append(linked_seg)

        n_unique = df_seg["frame"].nunique()
        if n_unique != frames_per_seg:
            logger.warning(
                "Segment %d: expected %d unique frames, found %d",
                seg_idx, frames_per_seg, n_unique,
            )
        logger.info(
            "Segment %d: frames %d-%d, particles %d",
            seg_idx, df_seg["frame"].min(), df_seg["frame"].max(), linked_seg["particle"].nunique(),
        )

    if linked_segments:
        F_linked = pd.concat(linked_segments, ignore_index=True)
    else:
        logger.warning("No segments with detections — returning empty DataFrame")
        F_linked = pd.DataFrame()

    b = cfg.boundary
    F_linked["boundary"] = (
        (F_linked.x < b.x_min) | (F_linked.x > b.x_max) |
        (F_linked.y < b.y_min) | (F_linked.y > b.y_max)
    )

    F_linked["rpx"] = F_linked.groupby("particle")["rpx"].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x
    )

    return F_linked


def compute_orientations(cfg: Config, F_linked: pd.DataFrame, H: np.ndarray, src_helpers) -> pd.DataFrame:
    """Compute particle rotation angles from blue/UV channel images."""
    F_linked = F_linked.copy()
    roi = cfg.roi_as_tuple()

    if cfg.tracking.sampling.skip_orientation:
        logger.info("skip_orientation=True — skipping angle computation.")
        F_linked["dir_x"] = np.nan
        F_linked["dir_y"] = np.nan
        F_linked["angle"] = np.nan
        return F_linked

    if "dir_x" not in F_linked.columns:
        F_linked["dir_x"] = np.nan
    if "dir_y" not in F_linked.columns:
        F_linked["dir_y"] = np.nan

    group_cols = ["trial", "frame"] if cfg.tracking.sampling.sample_log_frames else ["frame"]
    grouped = F_linked.groupby(group_cols)

    records_rot = []
    n_groups = len(grouped.groups)
    for idx, key in enumerate(grouped.groups, 1):
        print(f"\r  Computing orientations: frame {idx}/{n_groups}...", end='', flush=True)
        f = grouped.get_group(key)
        frame_orig = int(f["frame"].iloc[0])

        path = os.path.join(cfg.exp_dir, f"blue_{frame_orig:d}.png")
        I, err = src_helpers.read_image_with_retry(path)
        if I is None:
            logger.warning("Skipping orientation for frame %d (file not found): %s", frame_orig, err)
            continue
        I = camera_align(cv2.flip(I, 1), H)
        if I.shape[2] == 3:
            I = I[:, :, 0]
        I = I[roi[0]:roi[1], roi[2]:roi[3]]

        records_rot.extend(src_helpers.compute_frame_orientations(f, I))

    print()  # Newline after progress
    if records_rot:
        rot_df = pd.DataFrame(
            records_rot, columns=["_idx", "dir_x", "dir_y", "angle_R2"]
        ).set_index("_idx")
        F_linked.update(rot_df)

    F_linked = src_helpers.compute_continuous_angles(F_linked)
    return F_linked


def compute_g2(cfg: Config, F_linked: pd.DataFrame, src_helpers, visualize: bool = True) -> pd.DataFrame:
    """Compute G² (photoelastic stress proxy) for each particle."""
    F_linked = F_linked.copy()
    roi = cfg.roi_as_tuple()
    group_cols = ["trial", "frame"] if cfg.tracking.sampling.sample_log_frames else ["frame"]
    grouped = F_linked.groupby(group_cols)

    frame_keys = sorted(grouped.groups.keys())
    n_frames = len(frame_keys)
    for frame_num, key in enumerate(frame_keys, 1):
        print(f"\r  Computing G²: frame {frame_num}/{n_frames}...", end='', flush=True)
        frame_data = grouped.get_group(key)
        frame_orig = int(frame_data["frame"].iloc[0])

        image_path = os.path.join(cfg.exp_dir, f"bw_{frame_orig}.png")
        img, err = src_helpers.read_image_with_retry(image_path)
        if img is None:
            logger.warning("Could not load %s, skipping: %s", image_path, err)
            continue

        img = img[roi[0]:roi[1], roi[2]:roi[3]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 1).astype(np.float32)
        gray = gray / gray.max()
        gray = src_helpers.subtract_gaussian_rings(gray, frame_data, sigma_frac=cfg.g2_sigma_frac)

        G2_map = src_helpers.compute_G2_map(gray)
        h, w = G2_map.shape

        for idx, row in frame_data.iterrows():
            x = int(np.around(float(row["x"])))
            y = int(np.around(float(row["y"])))
            r = int(row["rpx"])
            y1, y2 = max(0, y - r), min(h, y + r)
            x1, x2 = max(0, x - r), min(w, x + r)
            if (y2 - y1) == 0 or (x2 - x1) == 0:
                F_linked.at[idx, "G2"] = np.nan
                continue
            G2_crop = src_helpers.crop_circle_with_mask_float(G2_map[y1:y2, x1:x2])
            F_linked.at[idx, "G2"] = float(np.sum(G2_crop[G2_crop > 0]))

    print()  # Newline after progress
    if visualize:
        src_helpers.visualize_g2(F_linked, cfg.data_dir, cfg.exp_folder, roi)

    return F_linked


def save_tracking(cfg: Config, F_linked: pd.DataFrame) -> str | None:
    """Save trajectory DataFrame to pickle."""
    if not cfg.tracking.save:
        logger.info("Skipping trajectory save (cfg.tracking.save=False)")
        return None
    
    out_path = cfg.output_path()
    F_linked.to_pickle(out_path)
    logger.info("Saved %d trajectory rows to %s", len(F_linked), out_path)
    return out_path


def run_tracking_pipeline(cfg: Config, src_helpers) -> pd.DataFrame:
    """Run full tracking pipeline: detect → link → orient → G² → save.
    
    Returns
    -------
    pd.DataFrame
        Trajectory DataFrame with all columns
    """
    configure_logging(cfg)
    cfg.validate()

    H = load_calibration(cfg.calibration_file)
    model = load_stardist_model(cfg.detection.model_name)

    max_frame_green = int(src_helpers.max_num(cfg.data_dir, cfg.exp_folder, "green_"))
    logger.info("Max green frame: %d", max_frame_green)

    frames_to_process, global_to_sample = build_frame_plan(cfg, max_frame_green)
    logger.info("Processing %d frames", len(frames_to_process))

    df_filtered = detect_disks(cfg, model, H, frames_to_process, global_to_sample, src_helpers)
    F_linked = link_trajectories(cfg, df_filtered)
    F_linked = compute_orientations(cfg, F_linked, H, src_helpers)
    F_linked = src_helpers.interpolate_pos_angle(F_linked)
    F_linked = compute_g2(cfg, F_linked, src_helpers)

    if cfg.tracking.verbose:
        logger.info("Visualizing random tracking frame...")
        roi = cfg.roi_as_tuple()
        src_helpers.visualize_detection(F_linked, cfg.data_dir, cfg.exp_folder, roi)

    save_tracking(cfg, F_linked)

    return F_linked
