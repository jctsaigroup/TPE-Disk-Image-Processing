"""
Contact detection pipeline: load trajectory → build bonds → ResNet18 inference → post-process.

Requires: new_torch_env (PyTorch)
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import pandas as pd

from config import Config, configure_logging

logger = logging.getLogger(__name__)


def load_tracking_result(cfg: Config) -> pd.DataFrame:
    """Load pre-computed tracking result from pickle."""
    pkl_path = os.path.join(cfg.pkl_dir, f"{cfg.exp_folder}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Tracking pickle not found: {pkl_path}")
    
    F_linked = pd.read_pickle(pkl_path)
    logger.info("Loaded %d trajectory rows from %s", len(F_linked), pkl_path)
    return F_linked


def get_frames_to_process(cfg: Config, all_frames: list) -> list:
    """Apply frame_selection config to filter frames for contact detection."""
    mode = cfg.contact.frame_selection.get("mode", "all")
    value = cfg.contact.frame_selection.get("value")
    
    if mode == "random" and value:
        import random
        frames_to_process = sorted(random.sample(all_frames, min(int(value), len(all_frames))))
    elif mode == "first" and value:
        frames_to_process = all_frames[:int(value)]
    elif mode == "single" and value:
        frames_to_process = [int(value)] if int(value) in all_frames else []
    else:
        frames_to_process = all_frames
    
    logger.info("Processing %d/%d frames (mode=%s, value=%s)", 
                len(frames_to_process), len(all_frames), mode, value)
    return frames_to_process


def detect_contacts(cfg: Config, F_linked: pd.DataFrame, src_helpers) -> pd.DataFrame:
    """Build candidate bonds, run ResNet18 inference, post-process singular bonds.
    
    Returns DataFrame with columns: i, j, xi, yi, xj, yj, ri, rj, prob, singular, frame
    """
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models

    roi = cfg.roi_as_tuple()
    d_tol = cfg.contact.d_tol
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg.contact.model_path
    
    _backbone = tv_models.resnet18(weights=None)
    _backbone.fc = nn.Sequential(
        nn.Linear(_backbone.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0),
        nn.Linear(256, 2)
    )
    _backbone.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = _backbone.to(device).eval()
    logger.info("Contact detection model loaded from %s", model_path)
    
    # Apply frame selection
    all_frames = sorted(F_linked['frame'].unique())
    frames_to_process = get_frames_to_process(cfg, all_frames)
    
    # Stage 1: Build candidate bonds
    grouped_F = F_linked.groupby('frame')
    F_bond = []
    n_frames = len(frames_to_process)
    
    for frame_idx, frame in enumerate(frames_to_process, 1):
        print(f"\r  Building bonds: frame {frame_idx}/{n_frames}...", end='', flush=True)
        if frame not in grouped_F.groups:
            continue
        f = grouped_F.get_group(frame).reset_index(drop=True)
        boundary_pid = f.particle[f.boundary.astype(bool)].to_numpy()
        
        F_bond_temp, *_ = src_helpers.get_all_bonds(f, boundary_pid, d_tol)
        F_bond_temp['frame'] = frame
        if 'log_sampled_frames' in f.columns:
            F_bond_temp['log_sampled_frame'] = f.log_sampled_frames.iloc[0]
        if 'trial' in f.columns:
            F_bond_temp['trial'] = f.trial.iloc[0]
        F_bond.append(F_bond_temp)
    
    if not F_bond:
        logger.warning("No candidate bonds produced — returning empty contact DataFrame")
        return pd.DataFrame()
    
    F_bond = pd.concat(F_bond, ignore_index=True)
    logger.info("Built %d candidate bonds", len(F_bond))
    
    print()  # Newline after progress
    
    # Stage 2: Run ResNet inference
    pred_frames = []
    grouped_bond = F_bond.groupby('frame')
    
    for frame_idx, frame in enumerate(frames_to_process, 1):
        print(f"\r  ResNet inference: frame {frame_idx}/{n_frames}...", end='', flush=True)
        if frame not in grouped_F.groups or frame not in grouped_bond.groups:
            continue
        
        PE_img_path = os.path.join(cfg.exp_dir, f'bw_{frame}.png')
        I = src_helpers.read_image_with_retry(PE_img_path)[0]
        if I is None:
            logger.warning("Skipping contact inference for frame %d (image not found)", frame)
            continue
        I = I[roi[0]:roi[1], roi[2]:roi[3]]
        if I.ndim == 3:
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        
        f_bond_frame = grouped_bond.get_group(frame).copy()
        preds, _ = src_helpers.predict_contact_batch(f_bond_frame, I, model, plot_raw=False, batch_size=32)
        
        f_bond_frame['contact'] = np.argmax(preds, axis=1)
        f_bond_frame['prob'] = np.max(preds, axis=1)
        pred_frames.append(f_bond_frame)
    
    if not pred_frames:
        logger.warning("No contact predictions produced — returning empty contact DataFrame")
        return pd.DataFrame()
    
    F_pred = pd.concat(pred_frames, ignore_index=True)
    F_pred = src_helpers.fill_temporal_single_frame_gaps(F_pred)
    logger.info("Contact inference complete")
    
    print()  # Newline after progress
    
    # Stage 3: Post-process singular bonds
    all_frames = []
    
    for frame_idx, frame in enumerate(frames_to_process, 1):
        print(f"\r  Post-processing: frame {frame_idx}/{n_frames}...", end='', flush=True)
        if frame not in grouped_F.groups:
            continue
        
        f = grouped_F.get_group(frame).copy()
        boundary_pid = f.particle[f.boundary.astype(bool)].to_numpy()
        
        f_bond_frame_full = F_pred[F_pred.frame == frame].copy()
        f_bond_frame = f_bond_frame_full.copy()
        
        # Keep confirmed contacts only
        f_bond_frame = f_bond_frame[f_bond_frame.contact > 0]
        if f_bond_frame.empty:
            continue
        
        # Mark / drop singular bonds
        f_bond_frame = src_helpers.process_singular_bonds(f_bond_frame, boundary_pid)
        
        # Promote best non-contact for each singular particle
        promoted = src_helpers.promote_singular_best_contact(f_bond_frame, f_bond_frame_full, boundary_pid)
        if not promoted.empty:
            f_bond_frame = pd.concat([f_bond_frame, promoted], ignore_index=True)
        
        # Duplicate bulk bonds so every particle has contacts listed under i
        f_bond_frame = src_helpers.duplicate_and_swap_bulk(f_bond_frame)
        
        all_frames.append(f_bond_frame)
    
    print()  # Newline after progress
    if not all_frames:
        logger.warning("No contacts after singular-bond filtering — returning empty contact DataFrame")
        return pd.DataFrame()
    
    F_contact = pd.concat(all_frames, ignore_index=True).drop(columns=['contact'])
    logger.info("Contact detection complete: %d contacts across %d frames", len(F_contact), F_contact['frame'].nunique())
    
    return F_contact


def save_contacts(cfg: Config, F_contact: pd.DataFrame) -> str | None:
    """Save contact bonds to pickle."""
    if not cfg.contact.save:
        logger.info("Skipping contact save (cfg.contact.save=False)")
        return None
    
    if F_contact.empty:
        logger.info("Skipping contact save (no contact data)")
        return None
    
    contact_path = os.path.join(cfg.bond_dir, f"CONTACT_BOND_{cfg.exp_folder}.pkl")
    F_contact.to_pickle(contact_path)
    logger.info("Saved %d contact bonds to %s", len(F_contact), contact_path)
    return contact_path


def run_contact_pipeline(cfg: Config, src_helpers) -> pd.DataFrame:
    """Run contact detection pipeline: load trajectory → detect contacts → save.
    
    Returns
    -------
    pd.DataFrame
        Contact bond DataFrame
    """
    configure_logging(cfg)
    cfg.validate()

    F_linked = load_tracking_result(cfg)
    F_contact = detect_contacts(cfg, F_linked, src_helpers)

    if not F_contact.empty and cfg.contact.verbose:
        logger.info("Visualizing random contact frame...")
        roi = cfg.roi_as_tuple()
        import random
        frame = random.choice(F_contact['frame'].unique())
        PE_img_path = os.path.join(cfg.exp_dir, f'bw_{frame}.png')
        I = src_helpers.read_image_with_retry(PE_img_path)[0]
        if I is not None:
            I = I[roi[0]:roi[1], roi[2]:roi[3]]
            if I.ndim == 3:
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            f_bonds = F_contact[F_contact.frame == frame]
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(src_helpers.plot_contacts(I, F_linked[F_linked.frame == frame], f_bonds, F_linked[F_linked.frame == frame].particle[F_linked[F_linked.frame == frame].boundary.astype(bool)].to_numpy()))
            plt.title(f"Contact bonds — Frame {frame}")
            plt.axis('off')
            plt.tight_layout()
            viz_dir = os.path.join(cfg.bond_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            output_path = os.path.join(viz_dir, f"{cfg.exp_folder}_frame_{frame}.png")
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info("Saved visualization to %s", output_path)

    save_contacts(cfg, F_contact)

    return F_contact
