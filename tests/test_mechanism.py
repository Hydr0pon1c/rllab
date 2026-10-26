from __future__ import annotations

import pytest

from rllab.config import GeneratorConfig
from rllab.experiment.corpus import (
    ExperimentCorpus,
    InstanceBundle,
    LabelingInfo,
)
from rllab.experiment.evaluation import PolicySnapshot
from rllab.experiment.mechanism import (
    aggregate_policy_update_decomposition,
    finite_policy_map_components,
    finite_set_shapley,
    finite_policy_response_comparison,
    finite_target_exposure_terms,
    policy_update_decomposition,
)
from rllab.mdp.generate import generate_layered_mdp
from rllab.mdp.solve import mdp_arrays, uniform_policy


def test_policy_update_decomposition_is_exact_and_signed():
    mdp = generate_layered_mdp(
        GeneratorConfig(horizon=2, width=4, terminal_width=4, num_actions=4),
        seed=113,
        mdp_id="decomposition-test",
    )
    arrays = mdp_arrays(mdp)
    before_policy = uniform_policy(mdp)
    after_policy = before_policy.copy()
    after_policy[~arrays.terminal, 0] += 0.01
    after_policy[~arrays.terminal, 1] -= 0.01
    bundle = InstanceBundle(
        mdp=mdp,
        split="train",
        labeling=LabelingInfo(
            labeling_id="test-labels",
            mdp_id=mdp.mdp_id,
            mode="random",
            state_token_ids={},
            action_token_ids=(32, 33, 34, 35),
            source="test",
        ),
        prompts={},
        truth={},
    )
    corpus = ExperimentCorpus(
        identity=None,  # type: ignore[arg-type]
        bundles=(bundle,),
        action_token_ids=(32, 33, 34, 35),
        prompt_length=94,
        labeling_mode="random",
        split="train",
    )
    result = policy_update_decomposition(
        corpus,
        PolicySnapshot({mdp.mdp_id: before_policy}),
        PolicySnapshot({mdp.mdp_id: after_policy}),
        exact_opportunity_by_mdp={mdp.mdp_id: 0.009},
        nominal_projection_by_mdp={mdp.mdp_id: 0.011},
        realized_projection_by_mdp={mdp.mdp_id: 0.010},
    )
    row = result["per_mdp"][mdp.mdp_id]
    reconstructed = sum(
        row[name]
        for name in (
            "exact_direction_opportunity",
            "nominal_projected_credit",
            "materialization",
            "policy_map_curvature",
            "occupancy_redistribution",
        )
    )
    assert reconstructed == pytest.approx(row["actual_gain"], abs=1.0e-14)
    assert row["nominal_projected_credit"] == pytest.approx(0.002)
    assert row["materialization"] == pytest.approx(-0.001)
    assert abs(row["performance_identity_residual"]) < 1.0e-12

    aggregate = aggregate_policy_update_decomposition(
        corpus,
        PolicySnapshot({mdp.mdp_id: before_policy}),
        PolicySnapshot({mdp.mdp_id: after_policy}),
        exact_opportunity=0.009,
        nominal_projection=0.011,
        realized_projection=0.010,
    )["aggregate"]
    assert aggregate["nominal_projected_credit"] == pytest.approx(0.002)
    assert aggregate["materialization"] == pytest.approx(-0.001)
    assert aggregate["additive_residual"] == pytest.approx(0.0, abs=1.0e-14)


def test_finite_policy_response_comparison_stratifies_a_scaled_response():
    mdp = generate_layered_mdp(
        GeneratorConfig(horizon=2, width=4, terminal_width=4, num_actions=4),
        seed=117,
        mdp_id="response-test",
    )
    arrays = mdp_arrays(mdp)
    before = uniform_policy(mdp)
    exact_after = before.copy()
    exact_after[~arrays.terminal, 0] += 0.02
    exact_after[~arrays.terminal, 1] -= 0.02
    sampled_after = before + 0.5 * (exact_after - before)
    bundle = InstanceBundle(
        mdp=mdp,
        split="train",
        labeling=LabelingInfo(
            labeling_id="test-labels",
            mdp_id=mdp.mdp_id,
            mode="random",
            state_token_ids={},
            action_token_ids=(32, 33, 34, 35),
            source="test",
        ),
        prompts={},
        truth={},
    )
    corpus = ExperimentCorpus(
        identity=None,  # type: ignore[arg-type]
        bundles=(bundle,),
        action_token_ids=(32, 33, 34, 35),
        prompt_length=94,
        labeling_mode="random",
        split="train",
    )
    current_state = next(
        state.state_id for state in mdp.ordered_states() if not state.terminal
    )
    result = finite_policy_response_comparison(
        corpus,
        PolicySnapshot({mdp.mdp_id: before}),
        PolicySnapshot({mdp.mdp_id: exact_after}),
        PolicySnapshot({mdp.mdp_id: sampled_after}),
        policy_loss_states={mdp.mdp_id: [current_state]},
    )
    assert result["aggregate"]["cosine"] == pytest.approx(1.0)
    assert result["aggregate"]["candidate_to_reference_l2_ratio"] == pytest.approx(0.5)
    assert result["aggregate"]["relative_l2_error"] == pytest.approx(0.5)
    assert "policy_loss_current" in result["exposure"]
    assert "destination_context_only" in result["exposure"]


def test_finite_target_exposure_terms_close_to_exact_gain():
    mdp = generate_layered_mdp(
        GeneratorConfig(horizon=3, width=4, terminal_width=4, num_actions=4),
        seed=119,
        mdp_id="target-exposure-test",
    )
    arrays = mdp_arrays(mdp)
    before = uniform_policy(mdp)
    after = before.copy()
    after[~arrays.terminal, 0] += 0.015
    after[~arrays.terminal, 1] -= 0.015
    bundle = InstanceBundle(
        mdp=mdp,
        split="train",
        labeling=LabelingInfo(
            labeling_id="test-labels",
            mdp_id=mdp.mdp_id,
            mode="random",
            state_token_ids={},
            action_token_ids=(32, 33, 34, 35),
            source="test",
        ),
        prompts={},
        truth={},
    )
    corpus = ExperimentCorpus(
        identity=None,  # type: ignore[arg-type]
        bundles=(bundle,),
        action_token_ids=(32, 33, 34, 35),
        prompt_length=94,
        labeling_mode="random",
        split="train",
    )
    root = mdp.start_state
    result = finite_target_exposure_terms(
        corpus,
        PolicySnapshot({mdp.mdp_id: before}),
        PolicySnapshot({mdp.mdp_id: after}),
        policy_loss_states={mdp.mdp_id: [root]},
    )
    groups = result["groups"]
    assert all(groups[name]["states"] > 0 for name in groups)
    assert sum(group["actual_gain"] for group in groups.values()) == pytest.approx(
        result["aggregate"]["actual_gain"], abs=1.0e-14
    )
    assert result["aggregate"]["identity_residual"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["aggregate"]["max_abs_per_mdp_identity_residual"] < 1.0e-12


def test_finite_set_shapley_separates_additive_and_joint_effects():
    weights = (1.0, 2.0, -3.0, 4.0)
    additive = {
        mask: sum(weight for index, weight in enumerate(weights) if mask & (1 << index))
        for mask in range(16)
    }
    result = finite_set_shapley(additive, ("b0", "b1", "b2", "b3"))
    assert [result["players"][f"b{i}"]["value"] for i in range(4)] == pytest.approx(
        weights
    )
    assert result["mobius_by_order"]["1"] == pytest.approx(sum(weights))
    assert result["mobius_by_order"]["2"] == pytest.approx(0.0)
    assert result["efficiency_residual"] == pytest.approx(0.0)

    pure_joint = {mask: float(mask == 15) for mask in range(16)}
    joint = finite_set_shapley(pure_joint, ("b0", "b1", "b2", "b3"))
    assert [joint["players"][f"b{i}"]["value"] for i in range(4)] == pytest.approx(
        [0.25] * 4
    )
    assert joint["mobius_by_order"]["4"] == pytest.approx(1.0)


def test_finite_policy_map_components_close_the_full_actual_gain():
    result = finite_policy_map_components(
        (
            {"actual_gain": 1.2, "fixed_occupancy_policy_gain": 1.0},
            {"actual_gain": 2.1, "fixed_occupancy_policy_gain": 2.0},
        ),
        additive_logit_fixed_gain=3.5,
        full_actual_gain=4.4,
        full_fixed_gain=4.0,
    )
    assert result["singleton"] == pytest.approx(3.3)
    assert result["softmax"] == pytest.approx(0.5)
    assert result["logit_map"] == pytest.approx(0.5)
    assert result["occupancy"] == pytest.approx(0.1)
    assert result["total"] == pytest.approx(4.4)
    assert result["identity_residual"] == pytest.approx(0.0)
