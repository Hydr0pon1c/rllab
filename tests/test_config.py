from __future__ import annotations

import pytest

from rllab.config import dataset_config_from_dict
from rllab.errors import ConfigurationError


def test_config_rejects_unknown_keys():
    with pytest.raises(ConfigurationError, match="unknown keys under generator"):
        dataset_config_from_dict({"generator": {"horizen": 8}})


def test_config_rejects_multi_digit_display_costs():
    with pytest.raises(ConfigurationError, match="max_cost must be <= 9"):
        dataset_config_from_dict({"generator": {"max_cost": 10}})


def test_one_instance_is_assigned_to_train():
    from rllab.artifacts import _split_assignments

    config = dataset_config_from_dict(
        {
            "dataset": {"num_instances": 1},
            "splits": {"train": 0.8, "validation": 0.1, "test": 0.1},
        }
    )
    assert _split_assignments(config) == ("train",)
