"""
Configuration loading and validation.
Shared by pipeline_tracking and pipeline_contact.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class AttrDict(dict):
    """Dict subclass allowing dot-notation attribute access."""
    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
            # Convert 0/1 to False/True for convenience
            if isinstance(val, int) and val in (0, 1):
                return bool(val)
            return val
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class Config:
    """Config wrapper for YAML-loaded configuration from dynamic and stationary files."""
    
    def __init__(self, yaml_dict: dict, stationary_dict: dict | None = None, config_path: str = None, stationary_path: str = None) -> None:
        """Initialize Config by merging dynamic and stationary YAML dicts.
        
        Parameters
        ----------
        yaml_dict : dict
            Loaded dynamic YAML (paths, experiment, tracking [sampling, frame_selection, verbose, save], logging)
        stationary_dict : dict, optional
            Loaded stationary YAML (roi, detection, linking, boundary, g2, contact). If None, assumes
            the sections are in yaml_dict.
        config_path : str, optional
            Path to the dynamic config file
        stationary_path : str, optional
            Path to the stationary config file
        """
        self._data = yaml_dict
        self._stationary_data = stationary_dict or {}
        self.config_path = config_path
        self.stationary_path = stationary_path
        
        # Merge configs: stationary takes precedence for overlapping keys
        merged = {**yaml_dict, **self._stationary_data}
        
        # Flatten nested paths to top level
        paths = yaml_dict.get("paths", {})
        experiment = yaml_dict.get("experiment", {})
        g2 = self._stationary_data.get("g2", {}) or yaml_dict.get("g2", {})
        
        self.data_dir = paths.get("data_dir")
        self.pkl_dir = paths.get("pkl_dir")
        self.bond_dir = paths.get("bond_dir")
        self.force_dir = paths.get("force_dir")
        self.calibration_file = paths.get("calibration_file")
        
        # exp_folder can be a string or list; store both
        exp_folder_raw = experiment.get("exp_folder")
        if isinstance(exp_folder_raw, list):
            self._exp_folder_list = exp_folder_raw
            self.exp_folder = exp_folder_raw[0] if exp_folder_raw else None
        else:
            self._exp_folder_list = [exp_folder_raw] if exp_folder_raw else []
            self.exp_folder = exp_folder_raw
        
        self.exp_dir = os.path.join(self.data_dir, self.exp_folder) if self.data_dir and self.exp_folder else self.data_dir
        self.g2_sigma_frac = g2.get("sigma_frac", 0.1)
        
        # Create nested objects with dot notation
        self.logging = AttrDict({
            "level": yaml_dict.get("logging", {}).get("level", "INFO"),
            "log_file": yaml_dict.get("logging", {}).get("log_file", None)
        })
        self.roi = AttrDict(self._stationary_data.get("roi", {}) or yaml_dict.get("roi", {}))
        self.detection = AttrDict(self._stationary_data.get("detection", {}) or yaml_dict.get("detection", {}))
        self.linking = AttrDict(self._stationary_data.get("linking", {}) or yaml_dict.get("linking", {}))
        self.boundary = AttrDict(self._stationary_data.get("boundary", {}) or yaml_dict.get("boundary", {}))
        
        # Tracking config: sampling and frame_selection grouped together
        tracking = yaml_dict.get("tracking", {})
        self.tracking = AttrDict({
            "sampling": AttrDict(tracking.get("sampling", {})),
            "frame_selection": AttrDict(tracking.get("frame_selection", {"mode": "all", "value": None})),
            "verbose": bool(tracking.get("verbose", False)),
            "save": bool(tracking.get("save", True))
        })
        
        # Legacy: also expose at top level for backward compatibility
        self.sampling = self.tracking.sampling
        self.frame_selection = self.tracking.frame_selection
        self.verbose_tracking = self.tracking.verbose
        self.save_tracking = self.tracking.save
        
        # Contact config: model_path and d_tol from stationary, verbose/save from dynamic
        contact_stationary = self._stationary_data.get("contact", {})
        contact_dynamic = yaml_dict.get("contact", {})
        self.contact = AttrDict({
            "model_path": contact_stationary.get("model_path", "models/contact_detect_best_model.pth"),
            "d_tol": contact_stationary.get("d_tol", 10),
            "verbose": bool(contact_dynamic.get("verbose", 0)),
            "save": bool(contact_dynamic.get("save", 0)),
            "frame_selection": AttrDict(contact_dynamic.get("frame_selection", {"mode": "all", "value": None}))
        })
        
        # Force config: hyperparameters from stationary, parallelism/frame_selection from dynamic
        force_stationary = self._stationary_data.get("force", {})
        force_dynamic = yaml_dict.get("force", {})
        self.force = AttrDict({
            "force_model": force_stationary.get("force_model", "models/force_model.pth"),
            "roi_batch_size": force_stationary.get("roi_batch_size", 256),
            "infer_batch_size": force_stationary.get("infer_batch_size", 32),
            "fsigma": force_stationary.get("fsigma", 1.0),
            "lr": force_stationary.get("lr", 1e-3),
            "n_iter": force_stationary.get("n_iter", 100),
            "tol": force_stationary.get("tol", 1e-4),
            "patience": force_stationary.get("patience", 10),
            "gpu_parallel_workers": force_dynamic.get("gpu_parallel_workers", 1),
            "plot_every": force_dynamic.get("plot_every", 0),
            "verbose": bool(force_dynamic.get("verbose", 0)),
            "frame_selection": AttrDict(force_dynamic.get("frame_selection", {"mode": "all", "value": None}))
        })
        
        # Handle both int (0/1) and bool (true/false) for verbose and save
        self.verbose_tracking = bool(yaml_dict.get("verbose_tracking", False))
        self.save_tracking = bool(yaml_dict.get("save_tracking", True))
    
    def roi_as_tuple(self) -> tuple:
        """Convert ROI dict to (y_min, y_max, x_min, x_max) tuple."""
        return (
            self.roi["y_min"],
            self.roi["y_max"],
            self.roi["x_min"],
            self.roi["x_max"],
        )
    
    def get_exp_folders(self) -> list[str]:
        """Get list of experiments to process."""
        return self._exp_folder_list
    
    def set_exp_folder(self, folder: str) -> None:
        """Set the current experiment folder (for iteration)."""
        self.exp_folder = folder
        self.exp_dir = os.path.join(self.data_dir, folder) if self.data_dir and folder else self.data_dir
    
    def output_path(self) -> str:
        """Return the output pickle file path."""
        return os.path.join(
            self.pkl_dir,
            f"{self.exp_folder}.pkl"
        )
    
    def validate(self) -> None:
        """Validate required config fields."""
        if self.exp_folder == "CHANGE_ME":
            raise ValueError(
                "exp_folder is set to 'CHANGE_ME' — must be configured before running"
            )


def load_config_from_yaml(yaml_path: str, stationary_path: str | None = None) -> tuple[Config, dict]:
    """Load configuration from dynamic and stationary YAML files.
    
    Parameters
    ----------
    yaml_path : str
        Path to the dynamic YAML config (paths, experiment, tracking [sampling, frame_selection, verbose, save], logging)
    stationary_path : str, optional
        Path to the stationary YAML config (roi, detection, linking, boundary, g2, contact).
        If None, looks for 'stationary_config.yml' in the same directory as yaml_path.
    
    Returns
    -------
    tuple[Config, dict]
        Config object and the loaded stationary dictionary
    """
    with open(yaml_path, encoding="utf-8") as f:
        dynamic_dict = yaml.safe_load(f)
    
    # Auto-detect stationary config if not provided
    if stationary_path is None:
        yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        stationary_path = os.path.join(yaml_dir, "stationary_config.yml")
    
    stationary_dict = {}
    if os.path.exists(stationary_path):
        with open(stationary_path, encoding="utf-8") as f:
            stationary_dict = yaml.safe_load(f) or {}
    
    return Config(dynamic_dict, stationary_dict, config_path=yaml_path, stationary_path=stationary_path), stationary_dict


def configure_logging(cfg: Config) -> None:
    """Configure logging from config."""
    handlers = [logging.StreamHandler()]
    if cfg.logging.log_file:
        handlers.append(logging.FileHandler(cfg.logging.log_file))
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
