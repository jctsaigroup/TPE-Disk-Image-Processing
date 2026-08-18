"""
Force vector solver pipeline: ResNet initial guess → GPU-optimized fitting → reciprocity symmetrization.

Workflow:
1. Crop contact ROIs and run ResNet18 → initial (force, angle) guesses
2. For each particle: optimize (force, alpha) to match photoelastic image
3. Validate reciprocity (ij vs ji), symmetrise using the better-fitting side
4. Save corrected force dataset

Requires: new_torch_env (PyTorch + CUDA)
"""

from __future__ import annotations

import logging
import os
import time
import concurrent.futures as cf

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from config import Config, configure_logging

logger = logging.getLogger(__name__)


def save_fit_plot(particle_id, images, output_dir):
    """Save particle fit comparison plot to output_dir.
    
    Parameters
    ----------
    particle_id : int
        Particle ID for filename
    images : dict
        Dictionary with keys: 'gray_img', 'guess_im', 'fit_im'
    output_dir : str
        Directory to save plot to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(6, 2))
    
    axes[0].imshow(images['gray_img'], cmap='gray', vmax=1)
    axes[0].set_title(f'id = {particle_id} \nexp', fontsize=10)
    axes[0].axis('off')
    
    axes[1].imshow(images['guess_im'], cmap='gray', vmax=1)
    axes[1].set_title('guess', fontsize=10)
    axes[1].axis('off')
    
    axes[2].imshow(images['fit_im'], cmap='gray', vmax=1)
    axes[2].set_title('fit', fontsize=10)
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'particle_{particle_id:06d}.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load trajectory and contact bond DataFrames."""
    traj_path = os.path.join(cfg.pkl_dir, f"{cfg.exp_folder}.pkl")
    bond_path = os.path.join(cfg.bond_dir, f"CONTACT_BOND_{cfg.exp_folder}.pkl")
    
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory pickle not found: {traj_path}")
    if not os.path.exists(bond_path):
        raise FileNotFoundError(f"Contact bond pickle not found: {bond_path}")
    
    F_traj = pd.read_pickle(traj_path)
    F_bond = pd.read_pickle(bond_path)
    
    logger.info("Loaded trajectory: %d rows", len(F_traj))
    logger.info("Loaded contact bonds: %d rows", len(F_bond))
    
    return F_traj, F_bond


def load_force_model(cfg: Config, src_helpers, stationary_cfg: dict) -> tuple[torch.nn.Module, torch.device]:
    """Load pre-trained ResNet18 for force/angle prediction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(__file__), stationary_cfg['force']['force_model'])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Force model not found: {model_path}")
    
    model = src_helpers.get_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    logger.info("Loaded force model from %s", model_path)
    logger.info("Using device: %s", device)
    
    return model, device


def make_resnet_predictions(cfg: Config, F_bond: pd.DataFrame, model: torch.nn.Module, 
                           device: torch.device, src_helpers, stationary_cfg: dict) -> pd.DataFrame:
    """
    Run ResNet18 on cropped contact ROIs to generate initial force/angle guesses.
    Returns DataFrame with force_pred and angle_pred columns appended.
    """
    # Define transforms (same as training)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    roi_batch_size = int(stationary_cfg['force']['roi_batch_size'])
    infer_batch_size = int(stationary_cfg['force']['infer_batch_size'])
    
    # Initialize output
    F_bond_pred = F_bond.copy()
    F_bond_pred['force_pred'] = np.nan
    F_bond_pred['angle_pred'] = np.nan
    
    total_contacts = len(F_bond_pred)
    logger.info("Starting ResNet prediction on %d contacts", total_contacts)
    logger.info("ROI batch size: %d, inference mini-batch: %d", roi_batch_size, infer_batch_size)
    
    with torch.no_grad():
        for chunk_images, chunk_meta, total_seen in src_helpers.iter_contact_roi_batches(
            F_bond, cfg.exp_dir, frame_lag=0, batch_size=roi_batch_size
        ):
            for start in range(0, len(chunk_images), infer_batch_size):
                end = min(start + infer_batch_size, len(chunk_images))
                batch_imgs = [data_transform(src_helpers.to_pil_uint8(im)) for im in chunk_images[start:end]]
                batch_tensor = torch.stack(batch_imgs).to(device)
                batch_predictions = model(batch_tensor).cpu().numpy()
                
                idxs = [m['idx'] for m in chunk_meta[start:end]]
                F_bond_pred.loc[idxs, 'force_pred'] = batch_predictions[:, 0]
                F_bond_pred.loc[idxs, 'angle_pred'] = batch_predictions[:, 1]
                
                del batch_tensor, batch_imgs, batch_predictions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            del chunk_images, chunk_meta
    
    logger.info("ResNet prediction complete")
    return F_bond_pred


def get_frames_to_process(cfg: Config, all_frames: list) -> list:
    """Apply frame_selection config to filter frames for force fitting."""
    mode = cfg.force.frame_selection.get("mode", "all")
    value = cfg.force.frame_selection.get("value")
    
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


def fit_forces(cfg: Config, F_bond_pred: pd.DataFrame, device: torch.device, 
              src_helpers, stationary_cfg: dict) -> pd.DataFrame:
    """
    Fit force/angle parameters for each contact by optimizing photoelastic match.
    Returns DataFrame with fitted force and alpha values.
    """
    F_bond_pred = src_helpers.append_ij_angle_to_pdata(F_bond_pred)
    F_bond_pred.force_pred = F_bond_pred.force_pred * 1.2  # Scale initial guess
    
    output_csv_path = os.path.join(cfg.force_dir, f'FORCE_FITTED_TEMP_{cfg.exp_folder}.csv')
    os.makedirs(cfg.force_dir, exist_ok=True)
    
    # Determine frames to process (support resume)
    all_frames = sorted(F_bond_pred['frame'].unique())
    frames_to_process = get_frames_to_process(cfg, all_frames)
    
    if os.path.exists(output_csv_path):
        try:
            existing = pd.read_csv(output_csv_path, usecols=['frame'])
            existing = existing.dropna(subset=['frame'])
            
            if len(existing) > 0:
                last_processed_frame = int(existing['frame'].max())
                frames_to_process = [f for f in all_frames if f > last_processed_frame]
                logger.info("Resuming from frame %d. %d frames remaining.", 
                           last_processed_frame, len(frames_to_process))
        except Exception as e:
            logger.warning("Could not read existing CSV (%s). Starting fresh.", e)
    
    if len(frames_to_process) == 0:
        logger.info("No new frames to process. Results up to date: %s", output_csv_path)
        df = pd.read_csv(output_csv_path)
        # Ensure numeric columns are properly typed
        numeric_cols = ['force', 'alpha', 'fitLoss', 'frame', 'i', 'j']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    if device.type != 'cuda':
        raise RuntimeError('Force fitting requires GPU (CUDA). Cannot proceed without GPU.')
    
    gpu_workers = cfg.force.gpu_parallel_workers
    plot_every = cfg.force.plot_every
    plots_dir = os.path.join(cfg.force_dir, 'fits') if plot_every else None
    
    # Prepare fit kwargs from stationary config - ensure proper numeric types
    fit_kwargs = dict(
        device=device,
        lr=float(stationary_cfg['force']['lr']),
        n_iter=int(stationary_cfg['force']['n_iter']),
        tol=float(stationary_cfg['force']['tol']),
        patience=int(stationary_cfg['force']['patience']),
        verbose=cfg.force.verbose,
        do_plot=bool(plot_every),  # Enable image generation when plotting is enabled
    )
    
    def run_fit(pid, pdata, img):
        return src_helpers.fit_one_particle_gpu(
            pid, pdata, img, float(stationary_cfg['force']['fsigma']),
            **fit_kwargs,
        )
    
    logger.info("Starting force fitting on %d frames with %d GPU workers", 
               len(frames_to_process), gpu_workers)
    if plots_dir:
        logger.info("Plot save directory: %s", plots_dir)
    
    for frame_idx, frame in enumerate(frames_to_process, 1):
        frame_t0 = time.perf_counter()
        logger.info("Frame %d/%d: %d", frame_idx, len(frames_to_process), frame)
        
        frame_data_out = []
        
        image_path = os.path.join(cfg.exp_dir, f'bw_{frame}.png')
        img, img_err = src_helpers.read_image_with_retry(image_path, retries=4, delay_s=0.15)
        if img is None:
            logger.warning("Skipped frame %d: image read failed (%s)", frame, img_err)
            continue
        
        frame_data = F_bond_pred[F_bond_pred['frame'] == frame]
        particle_groups = [(pid, pdata) for pid, pdata in frame_data.groupby('i') if len(pdata) > 1]
        n_candidates = len(particle_groups)
        n_failed = 0
        
        if n_candidates == 0:
            frame_dt = time.perf_counter() - frame_t0
            logger.info("Frame %d done: fitted=0/0, time=%.2fs", frame, frame_dt)
            continue
        
        if gpu_workers > 1:
            with cf.ThreadPoolExecutor(max_workers=gpu_workers) as ex:
                future_to_payload = {
                    ex.submit(run_fit, pid, pdata, img): (pid, pdata)
                    for pid, pdata in particle_groups
                }
                
                for fut in cf.as_completed(future_to_payload):
                    pid, pdata = future_to_payload[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        n_failed += 1
                        if cfg.force.verbose:
                            logger.warning("Worker failed at frame=%d, particle=%d: %s", frame, pid, e)
                        try:
                            res = run_fit(pid, pdata, img)
                        except Exception as e2:
                            if cfg.force.verbose:
                                logger.warning("Retry failed at frame=%d, particle=%d: %s", frame, pid, e2)
                            res = None
                    
                    if res is not None:
                        frame_data_out.append(res)
        else:
            for pid, pdata in particle_groups:
                try:
                    res = run_fit(pid, pdata, img)
                except Exception as e:
                    n_failed += 1
                    if cfg.force.verbose:
                        logger.warning("Sequential fit failed at frame=%d, particle=%d: %s", frame, pid, e)
                    res = None
                if res is not None:
                    frame_data_out.append(res)
        
        n_fitted = len(frame_data_out)
        
        if frame_data_out:
            for item in frame_data_out:
                pid, pdata_out, images = item
                if plots_dir and plot_every:
                    save_fit_plot(pid, images, plots_dir)
            
            frame_data_out.sort(key=lambda x: x[0])
            frame_results = pd.concat([x[1] for x in frame_data_out], ignore_index=True)
            
            write_header = (not os.path.exists(output_csv_path)) or os.path.getsize(output_csv_path) == 0
            frame_results.to_csv(output_csv_path, mode='a', header=write_header, index=False)
        
        frame_dt = time.perf_counter() - frame_t0
        logger.info("Frame done: fitted=%d/%d, failed=%d, time=%.2fs", 
                   n_fitted, n_candidates, n_failed, frame_dt)
    
    logger.info("Force fitting complete. Results saved to: %s", output_csv_path)
    
    if not os.path.exists(output_csv_path):
        logger.error("Output CSV not found at: %s. No successful fits were recorded.", output_csv_path)
        raise FileNotFoundError(f"Force fitting produced no results. Check fit convergence and output path: {output_csv_path}")
    
    df = pd.read_csv(output_csv_path)
    # Ensure numeric columns are properly typed
    numeric_cols = ['force', 'alpha', 'fitLoss', 'frame', 'i', 'j']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def symmetrize_forces(cfg: Config, F_bond_fitted: pd.DataFrame, src_helpers) -> pd.DataFrame:
    """
    Analyze reciprocity (ij vs ji) and symmetrize forces using the better-fitting side.
    Returns corrected force DataFrame.
    """
    logger.info("Analyzing reciprocity and symmetrizing forces")
    
    F_compare, F_bond_corrected, correction_stats = src_helpers.symmetrize_forces(F_bond_fitted)
    
    logger.info("Total contacts: %d", correction_stats['total_contacts'])
    logger.info("Contacts with reciprocal pairs corrected: %d", correction_stats['reciprocal_pairs'])
    logger.info("Contacts without reciprocal pairs (unchanged): %d", correction_stats['contacts_unchanged'])
    
    return F_bond_corrected


def save_results(cfg: Config, F_bond_corrected: pd.DataFrame):
    """Save final corrected force dataset to pickle."""
    os.makedirs(cfg.force_dir, exist_ok=True)
    
    out_filename = os.path.join(cfg.force_dir, f"{cfg.exp_folder}_Force_ResNet.pkl")
    F_bond_corrected.to_pickle(out_filename)
    logger.info("Saved final corrected results to: %s", out_filename)


def run_force_pipeline(cfg: Config, src_helpers, stationary_cfg: dict = None):
    """Main force vector solver pipeline."""
    configure_logging(cfg)
    
    # Use provided stationary config or extract from Config object
    if stationary_cfg is None:
        stationary_cfg = cfg._stationary_data
    
    logger.info("Starting force vector solver pipeline for: %s", cfg.exp_folder)
    
    # Load data
    F_traj, F_bond = load_data(cfg)
    
    # Initialize device (needed for fit_forces regardless of prediction cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Check if prediction pickle exists and is complete
    pred_path = os.path.join(cfg.force_dir, f'PREDICTION_{cfg.exp_folder}.pkl')
    os.makedirs(cfg.force_dir, exist_ok=True)
    
    F_bond_pred = None
    if os.path.exists(pred_path):
        try:
            F_bond_pred_cached = pd.read_pickle(pred_path)
            # Verify it has predictions for all contacts
            n_contacts = len(F_bond)
            n_with_preds = (F_bond_pred_cached['force_pred'].notna() & F_bond_pred_cached['angle_pred'].notna()).sum()
            
            if n_with_preds == n_contacts:
                logger.info("Found complete prediction cache with %d predictions. Skipping ResNet inference.", n_with_preds)
                # Ensure numeric columns are properly typed
                F_bond_pred_cached['force_pred'] = pd.to_numeric(F_bond_pred_cached['force_pred'], errors='coerce')
                F_bond_pred_cached['angle_pred'] = pd.to_numeric(F_bond_pred_cached['angle_pred'], errors='coerce')
                F_bond_pred = F_bond_pred_cached
            else:
                logger.info("Prediction cache incomplete: %d/%d predictions found. Running fresh inference.", n_with_preds, n_contacts)
        except Exception as e:
            logger.warning("Could not load prediction cache (%s). Running fresh inference.", e)
    
    if F_bond_pred is None:
        # Load model and make initial predictions
        model, device = load_force_model(cfg, src_helpers, stationary_cfg)
        F_bond_pred = make_resnet_predictions(cfg, F_bond, model, device, src_helpers, stationary_cfg)
        F_bond_pred.to_pickle(pred_path)
        logger.info("Saved ResNet predictions to: %s", pred_path)
    
    # Fit forces on GPU
    F_bond_fitted = fit_forces(cfg, F_bond_pred, device, src_helpers, stationary_cfg)
    
    # Symmetrize and correct forces
    F_bond_corrected = symmetrize_forces(cfg, F_bond_fitted, src_helpers)
    
    # Save final results
    save_results(cfg, F_bond_corrected)
    
    logger.info("Force vector solver pipeline complete!")
