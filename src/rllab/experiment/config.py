"""Strict configuration for reproducible PPU experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from rllab.errors import ConfigurationError


_CREDIT_MODES = {"orm", "exact_prm"}
_ATTENTION_IMPLEMENTATIONS = {"eager", "sdpa"}
_MASTER_DTYPES = {"float32"}
_COMPUTE_DTYPES = {"bfloat16", "float32"}
_SCALE_RULES = {"fixed_learning_rate", "fixed_parameter_step"}


@dataclass(frozen=True)
class ModelConfig:
    path: str = "models/Qwen3-8B"
    revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    attention_implementation: str = "sdpa"
    master_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (
                self.path,
                self.revision,
                self.attention_implementation,
                self.master_dtype,
                self.compute_dtype,
            )
        ) or not isinstance(self.gradient_checkpointing, bool):
            raise ConfigurationError("model fields have invalid types")
        if self.attention_implementation not in _ATTENTION_IMPLEMENTATIONS:
            raise ConfigurationError(
                "model.attention_implementation must be 'eager' or 'sdpa'"
            )
        if self.master_dtype not in _MASTER_DTYPES:
            raise ConfigurationError(
                "model.master_dtype must be float32; direct BF16 SGD loses small updates"
            )
        if self.compute_dtype not in _COMPUTE_DTYPES:
            raise ConfigurationError("model.compute_dtype must be bfloat16 or float32")
        if not self.revision:
            raise ConfigurationError("model.revision cannot be empty")


@dataclass(frozen=True)
class DataSelectionConfig:
    path: str = "data/study-v2-h32-w16-a4/latest"
    geometry_path: str | None = None
    split: str = "train"
    labeling_mode: str = "random"
    max_instances: int | None = None

    def validate(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (self.path, self.split, self.labeling_mode)
        ) or (
            self.geometry_path is not None and not isinstance(self.geometry_path, str)
        ):
            raise ConfigurationError("data fields have invalid types")
        if self.split not in {"train", "validation", "test"}:
            raise ConfigurationError("data.split must be train, validation, or test")
        if not self.labeling_mode:
            raise ConfigurationError("data.labeling_mode cannot be empty")
        if self.max_instances is not None and (
            not isinstance(self.max_instances, int)
            or isinstance(self.max_instances, bool)
            or self.max_instances < 1
        ):
            raise ConfigurationError("data.max_instances must be positive when set")


@dataclass(frozen=True)
class TrainingConfig:
    credit_mode: str = "exact_prm"
    steps: int = 100
    trajectories_global: int = 512
    train_microbatch_states: int = 32
    forward_batch_states: int = 64
    learning_rate: float = 1.0e-6
    scale_rule: str = "fixed_learning_rate"
    target_parameter_step_l2: float | None = None
    evaluation_interval: int = 5
    evaluation_steps: tuple[int, ...] = ()
    checkpoint_interval: int = 25
    keep_checkpoints: int = 1
    save_final_checkpoint: bool = True
    kernel_probe_steps: tuple[int, ...] = (0,)
    kernel_probe_fraction: float = 0.1

    def validate(self) -> None:
        if (
            not isinstance(self.credit_mode, str)
            or not isinstance(self.scale_rule, str)
            or not isinstance(self.save_final_checkpoint, bool)
        ):
            raise ConfigurationError("training fields have invalid types")
        if self.credit_mode not in _CREDIT_MODES:
            raise ConfigurationError(
                f"training.credit_mode must be one of {sorted(_CREDIT_MODES)}"
            )
        if self.scale_rule not in _SCALE_RULES:
            raise ConfigurationError(
                f"training.scale_rule must be one of {sorted(_SCALE_RULES)}"
            )
        positive_ints = (
            "steps",
            "trajectories_global",
            "train_microbatch_states",
            "forward_batch_states",
            "evaluation_interval",
            "checkpoint_interval",
            "keep_checkpoints",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigurationError(f"training.{name} must be positive")
        if (
            not isinstance(self.learning_rate, (int, float))
            or isinstance(self.learning_rate, bool)
            or not 0.0 < self.learning_rate
        ):
            raise ConfigurationError("training.learning_rate must be positive")
        if self.scale_rule == "fixed_parameter_step":
            if (
                not isinstance(self.target_parameter_step_l2, (int, float))
                or isinstance(self.target_parameter_step_l2, bool)
                or self.target_parameter_step_l2 <= 0.0
            ):
                raise ConfigurationError(
                    "training.target_parameter_step_l2 must be positive for "
                    "fixed_parameter_step"
                )
        elif self.target_parameter_step_l2 is not None:
            raise ConfigurationError(
                "training.target_parameter_step_l2 is only valid for "
                "fixed_parameter_step"
            )
        if (
            not isinstance(self.kernel_probe_fraction, (int, float))
            or isinstance(self.kernel_probe_fraction, bool)
            or not 0.0 < self.kernel_probe_fraction < 1.0
        ):
            raise ConfigurationError("training.kernel_probe_fraction must be in (0, 1)")
        if any(step < 0 or step >= self.steps for step in self.kernel_probe_steps):
            raise ConfigurationError(
                "training.kernel_probe_steps must fall in [0, training.steps)"
            )
        if tuple(sorted(set(self.kernel_probe_steps))) != self.kernel_probe_steps:
            raise ConfigurationError(
                "training.kernel_probe_steps must be sorted and contain no duplicates"
            )
        if any(step < 1 or step > self.steps for step in self.evaluation_steps):
            raise ConfigurationError(
                "training.evaluation_steps must fall in [1, training.steps]"
            )
        if tuple(sorted(set(self.evaluation_steps))) != self.evaluation_steps:
            raise ConfigurationError(
                "training.evaluation_steps must be sorted and contain no duplicates"
            )
        if self.evaluation_steps and any(
            step + 1 not in self.evaluation_steps for step in self.kernel_probe_steps
        ):
            raise ConfigurationError(
                "every training.kernel_probe_steps update must also appear in "
                "training.evaluation_steps"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 20260824
    deterministic_algorithms: bool = False
    allow_tf32: bool = False
    log_every: int = 1
    min_free_device_gib: float = 80.0

    def validate(self) -> None:
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ConfigurationError("runtime.seed cannot be negative")
        if (
            not isinstance(self.log_every, int)
            or isinstance(self.log_every, bool)
            or self.log_every < 1
        ):
            raise ConfigurationError("runtime.log_every must be positive")
        if (
            not isinstance(self.min_free_device_gib, (int, float))
            or isinstance(self.min_free_device_gib, bool)
            or self.min_free_device_gib <= 0
        ):
            raise ConfigurationError("runtime.min_free_device_gib must be positive")
        if not isinstance(self.deterministic_algorithms, bool) or not isinstance(
            self.allow_tf32, bool
        ):
            raise ConfigurationError("runtime boolean fields have invalid types")
        if self.allow_tf32:
            raise ConfigurationError(
                "runtime.allow_tf32 must remain false for the controlled update geometry"
            )


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs/puzzlemdp"
    run_name: str = "pilot"
    run_id: str | None = None
    resume: bool = False

    def validate(self) -> None:
        if (
            not isinstance(self.root, str)
            or not isinstance(self.run_name, str)
            or (self.run_id is not None and not isinstance(self.run_id, str))
        ):
            raise ConfigurationError("output path/name fields have invalid types")
        if not isinstance(self.resume, bool):
            raise ConfigurationError("output.resume must be boolean")
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for name, value in (("run_name", self.run_name), ("run_id", self.run_id)):
            if value is not None and (
                not value or any(char not in allowed for char in value)
            ):
                raise ConfigurationError(
                    f"output.{name} must contain only letters, numbers, '-' or '_'"
                )


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str = "rllab-experiment-config/v1"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataSelectionConfig = field(default_factory=DataSelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.schema_version != "rllab-experiment-config/v1":
            raise ConfigurationError(
                f"unsupported experiment schema_version: {self.schema_version!r}"
            )
        self.model.validate()
        self.data.validate()
        self.training.validate()
        self.runtime.validate()
        self.output.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_dataclass(cls: type, raw: Mapping[str, Any], path: str):
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ConfigurationError(f"unknown keys under {path}: {sorted(unknown)}")
    values = dict(raw)
    if cls is TrainingConfig:
        for name in ("kernel_probe_steps", "evaluation_steps"):
            if name not in values:
                continue
            steps = values[name]
            if not isinstance(steps, list) or not all(
                isinstance(value, int) for value in steps
            ):
                raise ConfigurationError(f"training.{name} must be a list of integers")
            values[name] = tuple(steps)
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigurationError(f"invalid values under {path}: {exc}") from exc


def experiment_config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    allowed = {"schema_version", "model", "data", "training", "runtime", "output"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigurationError(f"unknown experiment config keys: {sorted(unknown)}")
    for section in ("model", "data", "training", "runtime", "output"):
        if section in raw and not isinstance(raw[section], Mapping):
            raise ConfigurationError(
                f"experiment section {section!r} must be a mapping"
            )
    config = ExperimentConfig(
        schema_version=str(raw.get("schema_version", "rllab-experiment-config/v1")),
        model=_strict_dataclass(ModelConfig, raw.get("model", {}), "model"),
        data=_strict_dataclass(DataSelectionConfig, raw.get("data", {}), "data"),
        training=_strict_dataclass(TrainingConfig, raw.get("training", {}), "training"),
        runtime=_strict_dataclass(RuntimeConfig, raw.get("runtime", {}), "runtime"),
        output=_strict_dataclass(OutputConfig, raw.get("output", {}), "output"),
    )
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(
            f"cannot read experiment config {config_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError(
            f"experiment config {config_path} must be a YAML mapping"
        )
    return experiment_config_from_dict(raw)
