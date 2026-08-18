#!/usr/bin/env python
"""
run_tracking.py

Examples
--------
Single experiment, using config.yaml as-is:
    python run_tracking.py --config config.yaml

Override the experiment folder without touching the YAML (batch runs):
    python run_tracking.py --config config.yaml --exp-folder TPE_20260901A02_...

Batch over many experiments (bash):
    for f in TPE_20260808A01_... TPE_20260901A02_...; do
        python run_tracking.py --config config.yaml --exp-folder "$f"
    done

Dry run (validate config/paths without running the pipeline):
    python run_tracking.py --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from config import load_config_from_yaml
from pipeline_tracking import run_tracking_pipeline
import src as src_helpers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TPE disk tracking pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to dynamic config YAML")
    parser.add_argument("--stationary", default=None, help="Path to stationary config YAML (auto-detected if not provided)")
    parser.add_argument("--exp-folder", default=None, help="Override experiment.exp_folder from the config")
    parser.add_argument("--data-dir", default=None, help="Override paths.data_dir from the config")
    parser.add_argument("--pkl-dir", default=None, help="Override paths.pkl_dir from the config")
    parser.add_argument(
        "--no-log-frames", action="store_true",
        help="Force sample_log_frames=False (process every frame) regardless of config",
    )
    parser.add_argument(
        "--frame-mode", default=None,
        help="Override frame_selection.mode: 'all', 'random', 'first', or 'single'",
    )
    parser.add_argument(
        "--frame-value", type=int, default=None,
        help="Override frame_selection.value (depends on mode)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit without running")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, _ = load_config_from_yaml(args.config, args.stationary)

    # CLI overrides take precedence
    if args.exp_folder:
        cfg.set_exp_folder(args.exp_folder)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.pkl_dir:
        cfg.pkl_dir = args.pkl_dir
    
    if args.no_log_frames:
        cfg.tracking.sampling["sample_log_frames"] = False
    if args.frame_mode:
        cfg.tracking.frame_selection["mode"] = args.frame_mode
    if args.frame_value is not None:
        cfg.tracking.frame_selection["value"] = args.frame_value

    exp_folders = cfg.get_exp_folders()
    if not exp_folders:
        print("Error: No experiments configured")
        return 1

    if args.dry_run:
        cfg.validate()
        print(f"✓ Config valid. Would process {len(exp_folders)} experiment(s):")
        for folder in exp_folders:
            print(f"  - {folder}")
        return 0

    for folder in exp_folders:
        cfg.set_exp_folder(folder)
        print(f"\n{'='*60}")
        print(f"Processing: {folder}")
        print(f"{'='*60}")
        run_tracking_pipeline(cfg, src_helpers)

    print(f"\n✓ Completed {len(exp_folders)} experiment(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())