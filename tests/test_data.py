from __future__ import annotations

import unicodedata

import numpy as np

from rllab.config import GeneratorConfig
from rllab.data.calibration import (
    build_calibration_records,
    masked_softmax,
    uniformizing_fixed_bias,
)
from rllab.data.credit import trajectory_credit
from rllab.data.exposure import ExposureClass, classify_state_exposure
from rllab.data.prompts import build_all_prompt_records
from rllab.data.relabel import make_labeling, make_score_aligned_labeling
from rllab.data.trajectory import sample_trajectory
from rllab.mdp.generate import generate_layered_mdp
from rllab.mdp.environment import PuzzleEnv
from rllab.mdp.solve import uniform_policy


def tiny_mdp():
    return generate_layered_mdp(
        GeneratorConfig(horizon=3, width=4, terminal_width=4, num_actions=4),
        seed=71,
        mdp_id="tiny",
    )


def test_exposure_partition_has_required_three_classes():
    mdp = tiny_mdp()
    records = classify_state_exposure(mdp, [mdp.start_state])
    by_state = {record.state_id: record for record in records}
    assert by_state[mdp.start_state].exposure_class is ExposureClass.POLICY_LOSS_CURRENT
    destinations = {transition.target for transition in mdp.outgoing()[mdp.start_state]}
    assert all(
        by_state[state_id].exposure_class is ExposureClass.DESTINATION_CONTEXT_ONLY
        for state_id in destinations
    )
    assert any(
        record.exposure_class is ExposureClass.TOKEN_ABSENT for record in records
    )
    assert len(records) == len(mdp.states)


def test_uniformizing_bias_uses_the_same_masked_softmax():
    logits = np.asarray(
        [[4.0, -2.0, 0.5, 9.0], [-100.0, 3.0, 2.0, 1.0]], dtype=np.float32
    )
    bias = uniformizing_fixed_bias(logits)
    probabilities = masked_softmax(
        logits, np.ones_like(logits, dtype=bool), fixed_bias=bias
    )
    np.testing.assert_allclose(probabilities, 0.25, atol=1e-7)
    assert bias.dtype == np.float32
    records = build_calibration_records(
        model_fingerprint="model",
        token_registry_fingerprint="registry",
        mdp_id="mdp",
        labeling_id="labels",
        state_ids=("s0", "s1"),
        prompt_sha256=("p0", "p1"),
        action_token_ids=(32, 33, 34, 35),
        initial_action_logits=logits,
    )
    assert len(records) == 2
    assert records[0].fixed_bias_fp32 == tuple(float(value) for value in bias[0])


def test_sampled_trajectory_has_exact_length_and_terminal_cost():
    mdp = tiny_mdp()
    trajectory = sample_trajectory(mdp, uniform_policy(mdp), seed=83)
    assert len(trajectory.steps) == mdp.horizon
    assert (
        trajectory.total_cost == trajectory.transition_cost + trajectory.terminal_cost
    )
    assert next(
        state for state in mdp.states if state.state_id == trajectory.terminal_state
    ).terminal
    credit = trajectory_credit(mdp, trajectory, uniform_policy(mdp))
    assert credit.total_return == -trajectory.total_cost
    assert credit.steps[0].realised_return_to_go == credit.total_return
    assert credit.steps[-1].environment_reward == -(
        trajectory.steps[-1].immediate_cost + trajectory.terminal_cost
    )


def test_environment_snapshot_restore_and_counterfactuals():
    mdp = tiny_mdp()
    env = PuzzleEnv(mdp)
    root_snapshot = env.snapshot()
    counterfactuals = env.counterfactuals()
    assert len(counterfactuals) == mdp.num_actions
    result = env.step(2)
    assert result == counterfactuals[2]
    assert env.state_id == result.next_state_id
    env.restore(root_snapshot)
    assert env.state_id == mdp.start_state
    assert env.total_cost == 0


def test_prompt_token_contract_with_real_qwen(qwen_tokenizer, qwen_registry):
    mdp = tiny_mdp()
    labeling = make_labeling(mdp, qwen_registry, mode="random", seed=97)
    records = build_all_prompt_records(
        mdp,
        labeling,
        qwen_registry,
        qwen_tokenizer,
        enable_thinking=False,
    )
    assert len(records) == sum(not state.terminal for state in mdp.states)
    assert {len(record.input_ids) for record in records} == {94}
    assert all(
        record.allowed_action_token_ids == (32, 33, 34, 35) for record in records
    )
    for token in labeling.state_tokens.values():
        assert unicodedata.name(token.text, "").startswith("CJK UNIFIED IDEOGRAPH")
    for record in records:
        assert sum(slot.role == "current_state" for slot in record.slots) == 1
        assert sum(slot.role == "action" for slot in record.slots) == 4
        assert sum(slot.role == "destination_state" for slot in record.slots) == 4


def test_measured_score_assignment_supports_both_extremes(qwen_registry):
    mdp = tiny_mdp()
    size = len(mdp.states)
    scores = np.eye(size, dtype=np.float64)
    aligned = make_score_aligned_labeling(
        mdp, qwen_registry, scores, mode="aligned", seed=1
    )
    adversarial = make_score_aligned_labeling(
        mdp, qwen_registry, scores, mode="adversarial", seed=1
    )
    assert aligned.assignment_score == size
    assert adversarial.assignment_score == 0.0
    assert (
        aligned.assignment_metadata["score_matrix_sha256"]
        == adversarial.assignment_metadata["score_matrix_sha256"]
    )
    assert set(token.token_id for token in aligned.state_tokens.values()) == set(
        token.token_id for token in adversarial.state_tokens.values()
    )


def test_measured_score_assignment_can_hold_depth_token_pools_fixed(qwen_registry):
    mdp = tiny_mdp()
    states = mdp.ordered_states()
    blocks = tuple(
        (
            tuple(index for index, state in enumerate(states) if state.depth == depth),
            tuple(index for index, state in enumerate(states) if state.depth == depth),
        )
        for depth in range(mdp.horizon + 1)
    )
    scores = np.eye(len(states), dtype=np.float64)
    aligned = make_score_aligned_labeling(
        mdp,
        qwen_registry,
        scores,
        mode="aligned",
        seed=1,
        assignment_blocks=blocks,
    )
    adversarial = make_score_aligned_labeling(
        mdp,
        qwen_registry,
        scores,
        mode="adversarial",
        seed=1,
        assignment_blocks=blocks,
    )
    assert aligned.assignment_score == len(states)
    assert adversarial.assignment_score == 1.0
    assert aligned.assignment_metadata["assignment_block_sizes"] == [1, 4, 4, 4]
    for rows, columns in blocks:
        expected = {qwen_registry.state_tokens[column].token_id for column in columns}
        assert {
            aligned.state_tokens[states[row].state_id].token_id for row in rows
        } == expected
        assert {
            adversarial.state_tokens[states[row].state_id].token_id for row in rows
        } == expected
