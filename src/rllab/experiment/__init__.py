"""PPU experiment runtime for PuzzleMDP policy-gradient studies.

The package keeps imports light: CPU data tooling can import :mod:`rllab`
without importing PyTorch or the vendor accelerator stack.
"""

from rllab.experiment.config import ExperimentConfig, load_experiment_config

__all__ = ["ExperimentConfig", "load_experiment_config"]
