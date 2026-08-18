#!/usr/bin/env python
"""
run_contact.py

Contact detection pipeline: load pre-computed trajectory and run ResNet18 contact inference.

Examples
--------
Run contact detection on existing trajectory:
    python run_contact.py --config config.yaml

Override experiment folder:
    python run_contact.py --config config.yaml --exp-folder TPE_20260808A01_...

Dry run:
    python run_contact.py --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from config import load_config_from_yaml
from pipeline_contact import run_contact_pipeline
import src as src_helpers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contact detection on pre-computed trajectory.")
    parser.add_argument("--config", default="config.yaml", help="Path to dynamic config YAML")
    parser.add_argument("--stationary", default=None, help="Path to stationary config YAML (auto-detected if not provided)")
    parser.add_argument("--exp-folder", default=None, help="Override experiment.exp_folder from the config")
    parser.add_argument("--pkl-dir", default=None, help="Override paths.pkl_dir from the config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit without running")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, _ = load_config_from_yaml(args.config, args.stationary)

    # CLI overrides take precedence
    if args.exp_folder:
        cfg.set_exp_folder(args.exp_folder)
    if args.pkl_dir:
        cfg.pkl_dir = args.pkl_dir

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
        run_contact_pipeline(cfg, src_helpers)

    print(f"\n✓ Completed {len(exp_folders)} experiment(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
