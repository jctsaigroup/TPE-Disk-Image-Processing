#!/usr/bin/env python
"""
run_force.py

Force vector solver: ResNet initial guess → GPU-optimized fitting → reciprocity symmetrization.

Uses pre-computed trajectory and contact bonds to fit force/angle parameters by minimizing
the difference between synthetic and experimental photoelastic images.

Static config (stationary_config.yml):
- force.force_model: ResNet18 weights path
- force.roi_batch_size, force.infer_batch_size: Batching parameters
- force.fsigma: Photoelastic constant
- force hyperparameters: tol, patience, lr, n_iter

Dynamic config (dynamic_config.yaml):
- force.gpu_parallel_workers: GPU parallelism
- force.frame_selection: Which frames to process
- paths.force_dir: Output directory

Examples
--------
Run force solver on existing trajectory and contacts:
    python run_force.py --config dynamic_config.yaml

Override experiment folder:
    python run_force.py --config dynamic_config.yaml --exp-folder TPE_20260521A01_...

Process only frame 1401 with 4 GPU workers:
    python run_force.py --config dynamic_config.yaml --exp-folder TPE_20260521A01_... \\
        --force-frame-mode single --force-frame-value 1401 --gpu-workers 4

Dry run to validate config:
    python run_force.py --config dynamic_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config_from_yaml
from pipeline_force import run_force_pipeline
import src


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run force vector solver on pre-computed trajectory and contacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="dynamic_config.yaml", 
                       help="Path to dynamic config YAML")
    parser.add_argument("--stationary", default=None, 
                       help="Path to stationary config YAML (auto-detected if not provided)")
    parser.add_argument("--exp-folder", default=None, 
                       help="Override experiment.exp_folder from the config")
    parser.add_argument("--pkl-dir", default=None, 
                       help="Override paths.pkl_dir from the config")
    parser.add_argument("--bond-dir", default=None, 
                       help="Override paths.bond_dir from the config")
    parser.add_argument("--force-frame-mode", default=None, 
                       choices=["all", "random", "first", "single"],
                       help="Override force.frame_selection.mode from the config")
    parser.add_argument("--force-frame-value", default=None, type=int,
                       help="Override force.frame_selection.value from the config")
    parser.add_argument("--gpu-workers", default=None, type=int,
                       help="Override force.gpu_parallel_workers from the config")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Validate config and exit without running")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, _ = load_config_from_yaml(args.config, args.stationary)

    # CLI overrides take precedence
    if args.exp_folder:
        cfg.set_exp_folder(args.exp_folder)
    if args.pkl_dir:
        cfg.pkl_dir = args.pkl_dir
    if args.bond_dir:
        cfg.bond_dir = args.bond_dir
    if args.force_frame_mode:
        cfg.force.frame_selection["mode"] = args.force_frame_mode
    if args.force_frame_value is not None:
        cfg.force.frame_selection["value"] = args.force_frame_value
    if args.gpu_workers is not None:
        cfg.force.gpu_parallel_workers = args.gpu_workers

    exp_folders = cfg.get_exp_folders()
    if not exp_folders:
        print("Error: No experiments configured")
        return 1

    if args.dry_run:
        cfg.validate()
        print(f"✓ Config valid. Would process {len(exp_folders)} experiment(s):")
        for folder in exp_folders:
            print(f"  - {folder}")
        print("\nForce solver configuration:")
        print(f"  GPU workers: {cfg.force.gpu_parallel_workers}")
        print(f"  Frame mode: {cfg.force.frame_selection.get('mode', 'all')}")
        print(f"  Output dir: {cfg.force_dir}")
        return 0

    for folder in exp_folders:
        cfg.set_exp_folder(folder)
        print(f"\n{'='*70}")
        print(f"Force Solver: {folder}")
        print(f"{'='*70}")
        run_force_pipeline(cfg, src)

    print(f"\n✓ Completed {len(exp_folders)} experiment(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
