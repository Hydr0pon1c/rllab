"""Typed configuration loading for deterministic data builds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from rllab.errors import ConfigurationError


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for a layered deterministic PuzzleMDP.

    Every episode has exactly ``horizon`` transitions. Width grows only as fast
    as the action capacity permits, so all generated states remain reachable.
    """

    horizon: int = 8
    width: int = 16
    terminal_width: int | None = None
    num_actions: int = 4
    min_cost: int = 1
    max_cost: int = 9
    terminal_cost_min: int = 0
    terminal_cost_max: int = 20
    min_optimality_gap: int = 1

    def validate(self) -> None:
        if self.horizon < 1:
            raise ConfigurationError("generator.horizon must be at least 1")
        if self.width < 1:
            raise ConfigurationError("generator.width must be positive")
        if self.num_actions < 2:
            raise ConfigurationError("generator.num_actions must be at least 2")
        if self.min_cost < 0 or self.max_cost < self.min_cost:
            raise ConfigurationError(
                "generator costs must satisfy 0 <= min_cost <= max_cost"
            )
        if self.max_cost > 9:
            raise ConfigurationError(
                "generator.max_cost must be <= 9 in schema v1 so every displayed cost "
                "has the same one-token numeric layout"
            )
        if (
            self.terminal_cost_min < 0
            or self.terminal_cost_max < self.terminal_cost_min
        ):
            raise ConfigurationError(
                "terminal costs must satisfy 0 <= terminal_cost_min <= terminal_cost_max"
            )
        if self.min_optimality_gap < 0:
            raise ConfigurationError("generator.min_optimality_gap cannot be negative")
        if self.terminal_width is not None and self.terminal_width < 1:
            raise ConfigurationError(
                "generator.terminal_width must be positive when set"
            )
        terminal_width = (
            self.terminal_width if self.terminal_width is not None else self.width
        )
        if terminal_width < self.num_actions:
            raise ConfigurationError(
                "generator.terminal_width must be >= num_actions to avoid duplicate destinations"
            )


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def validate(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0 for value in values):
            raise ConfigurationError("split fractions cannot be negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ConfigurationError("split fractions must sum to 1")


@dataclass(frozen=True)
class DifficultyConfig:
    """Acceptance bands for task synthesis.

    ``None`` disables a bound. Metrics are measured after exact solution, so
    these constraints describe realised tasks rather than generator proxies.
    """

    min_delayed_choice_rate: float | None = 0.15
    max_delayed_choice_rate: float | None = 0.85
    min_uniform_regret_per_step: float | None = 0.10
    max_uniform_regret_per_step: float | None = 8.0
    min_mean_q_gap: float | None = 1.0
    max_mean_q_gap: float | None = 12.0
    min_mean_depth_entropy: float | None = 0.55
    max_generation_attempts: int = 256

    def validate(self) -> None:
        pairs = (
            (
                "delayed_choice_rate",
                self.min_delayed_choice_rate,
                self.max_delayed_choice_rate,
            ),
            (
                "uniform_regret_per_step",
                self.min_uniform_regret_per_step,
                self.max_uniform_regret_per_step,
            ),
            ("mean_q_gap", self.min_mean_q_gap, self.max_mean_q_gap),
        )
        for name, lower, upper in pairs:
            if lower is not None and upper is not None and lower > upper:
                raise ConfigurationError(f"difficulty bounds for {name} are reversed")
        for name in ("min_delayed_choice_rate", "max_delayed_choice_rate"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"difficulty.{name} must be in [0, 1]")
        if (
            self.min_mean_depth_entropy is not None
            and not 0.0 <= self.min_mean_depth_entropy <= 1.0
        ):
            raise ConfigurationError(
                "difficulty.min_mean_depth_entropy must be in [0, 1]"
            )
        if self.max_generation_attempts < 1:
            raise ConfigurationError(
                "difficulty.max_generation_attempts must be positive"
            )


@dataclass(frozen=True)
class TokenizerConfig:
    path: str = "models/Qwen3-8B"
    source: str = "Qwen/Qwen3-8B"
    revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    action_labels: tuple[str, ...] = ("A", "B", "C", "D")
    state_pool_size: int = 512
    enable_thinking: bool = False

    def validate(self, num_actions: int) -> None:
        if len(self.action_labels) != num_actions:
            raise ConfigurationError(
                f"tokenizer.action_labels has {len(self.action_labels)} entries, "
                f"but generator.num_actions is {num_actions}"
            )
        if len(set(self.action_labels)) != len(self.action_labels):
            raise ConfigurationError("tokenizer.action_labels must be unique")
        if self.state_pool_size < 1:
            raise ConfigurationError("tokenizer.state_pool_size must be positive")


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "pilot"
    seed: int = 20260824
    num_instances: int = 1
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    labeling_modes: tuple[str, ...] = ("random",)

    def validate(self) -> None:
        allowed_name_chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        )
        if not self.name or any(ch not in allowed_name_chars for ch in self.name):
            raise ConfigurationError(
                "dataset.name must be non-empty and contain only letters, numbers, '-' or '_'"
            )
        if self.num_instances < 1:
            raise ConfigurationError("dataset.num_instances must be positive")
        self.generator.validate()
        self.difficulty.validate()
        self.splits.validate()
        self.tokenizer.validate(self.generator.num_actions)
        allowed_modes = {"identity", "random"}
        invalid = set(self.labeling_modes) - allowed_modes
        if invalid:
            raise ConfigurationError(
                "Config-driven builds currently accept identity/random labelings; "
                f"geometry-driven modes require measured matrices via the Python API: {sorted(invalid)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_dataclass(cls: type, values: Mapping[str, Any], path: str):
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ConfigurationError(f"unknown keys under {path}: {sorted(unknown)}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigurationError(f"invalid values under {path}: {exc}") from exc


def dataset_config_from_dict(raw: Mapping[str, Any]) -> DatasetConfig:
    """Construct a config while rejecting misspelled/unknown keys."""

    allowed = {"dataset", "generator", "difficulty", "splits", "tokenizer", "labelings"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigurationError(f"unknown top-level config keys: {sorted(unknown)}")

    dataset_values = dict(raw.get("dataset", {}))
    dataset_known = {"name", "seed", "num_instances"}
    dataset_unknown = set(dataset_values) - dataset_known
    if dataset_unknown:
        raise ConfigurationError(
            f"unknown keys under dataset: {sorted(dataset_unknown)}"
        )

    generator = _strict_dataclass(
        GeneratorConfig, raw.get("generator", {}), "generator"
    )
    difficulty = _strict_dataclass(
        DifficultyConfig, raw.get("difficulty", {}), "difficulty"
    )
    splits = _strict_dataclass(SplitConfig, raw.get("splits", {}), "splits")

    tokenizer_values = dict(raw.get("tokenizer", {}))
    if "action_labels" in tokenizer_values:
        tokenizer_values["action_labels"] = tuple(tokenizer_values["action_labels"])
    tokenizer = _strict_dataclass(TokenizerConfig, tokenizer_values, "tokenizer")

    labeling_values = raw.get("labelings", ["random"])
    if not isinstance(labeling_values, list) or not all(
        isinstance(v, str) for v in labeling_values
    ):
        raise ConfigurationError("labelings must be a list of mode names")

    config = DatasetConfig(
        **dataset_values,
        generator=generator,
        difficulty=difficulty,
        splits=splits,
        tokenizer=tokenizer,
        labeling_modes=tuple(labeling_values),
    )
    config.validate()
    return config


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"config {config_path} must contain a YAML mapping")
    return dataset_config_from_dict(raw)
