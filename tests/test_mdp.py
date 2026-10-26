from __future__ import annotations

from itertools import product

import numpy as np

from rllab.config import DifficultyConfig, GeneratorConfig
from rllab.mdp.difficulty import measure_difficulty, synthesize_mdp
from rllab.mdp.generate import generate_layered_mdp
from rllab.mdp.schema import PuzzleMDP, validate_mdp
from rllab.mdp.solve import (
    evaluate_policy,
    mdp_arrays,
    optimal_policy,
    performance_difference,
    uniform_policy,
)


def small_config() -> GeneratorConfig:
    return GeneratorConfig(horizon=4, width=8, terminal_width=8, num_actions=4)


def test_generator_is_reproducible_reachable_and_balanced():
    left = generate_layered_mdp(small_config(), seed=17, mdp_id="same")
    right = generate_layered_mdp(small_config(), seed=17, mdp_id="same")
    assert left.to_dict() == right.to_dict()
    validate_mdp(left)
    counts = left.metadata["optimal_action_counts"]
    assert max(counts) - min(counts) <= 1
    assert any(state.terminal_cost > 0 for state in left.states if state.terminal)
    assert PuzzleMDP.from_dict(left.to_dict()).to_dict() == left.to_dict()


def test_exact_values_and_performance_difference_identity():
    mdp = generate_layered_mdp(small_config(), seed=29, mdp_id="exact")
    uniform = uniform_policy(mdp)
    optimal = optimal_policy(mdp)
    uniform_eval = evaluate_policy(mdp, uniform)
    optimal_eval = evaluate_policy(mdp, optimal)
    assert optimal_eval.objective >= uniform_eval.objective
    for state_index, state in enumerate(mdp.ordered_states()):
        if state.terminal:
            assert uniform_eval.value[state_index] == -state.terminal_cost

    identity = performance_difference(mdp, uniform, optimal)
    np.testing.assert_allclose(
        identity.exact_delta_j, identity.advantage_projection, atol=1e-12
    )
    np.testing.assert_allclose(identity.residual, 0.0, atol=1e-12)


def test_logit_policy_gradient_matches_finite_difference():
    mdp = generate_layered_mdp(small_config(), seed=41, mdp_id="gradient")
    arrays = mdp_arrays(mdp)
    policy = uniform_policy(mdp)
    evaluation = evaluate_policy(mdp, policy)
    state_index = next(
        index
        for index in range(len(arrays.state_ids))
        if not arrays.terminal[index] and evaluation.occupancy[index] > 0
    )
    action = 0
    epsilon = 1e-5

    def perturbed(sign: float):
        logits = np.zeros(mdp.num_actions, dtype=np.float64)
        logits[action] = sign * epsilon
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        result = policy.copy()
        result[state_index] = probabilities
        return result

    plus = evaluate_policy(mdp, perturbed(+1.0)).objective
    minus = evaluate_policy(mdp, perturbed(-1.0)).objective
    finite_difference = (plus - minus) / (2 * epsilon)
    assert np.isclose(
        finite_difference,
        evaluation.logit_gradient[state_index, action],
        atol=1e-8,
        rtol=1e-6,
    )


def test_enumerated_orm_and_exact_prm_have_the_true_expected_gradient():
    """Catch sign, baseline, horizon-sum, and per-step credit mistakes."""

    mdp = generate_layered_mdp(
        GeneratorConfig(horizon=3, width=4, terminal_width=4, num_actions=4),
        seed=43,
        mdp_id="credit-gradient",
    )
    arrays = mdp_arrays(mdp)
    rng = np.random.default_rng(47)
    policy = uniform_policy(mdp)
    logits = rng.normal(size=(int((~arrays.terminal).sum()), mdp.num_actions))
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    policy[~arrays.terminal] = probabilities / probabilities.sum(axis=1, keepdims=True)
    evaluation = evaluate_policy(mdp, policy)
    outgoing = mdp.outgoing()
    states = {state.state_id: state for state in mdp.states}

    estimates = {
        "orm": np.zeros_like(evaluation.logit_gradient),
        "exact_prm": np.zeros_like(evaluation.logit_gradient),
    }
    for actions in product(range(mdp.num_actions), repeat=mdp.horizon):
        state_id = mdp.start_state
        probability = 1.0
        transition_cost = 0
        path: list[tuple[int, int]] = []
        for action in actions:
            state_index = arrays.state_index[state_id]
            probability *= policy[state_index, action]
            path.append((state_index, action))
            transition = outgoing[state_id][action]
            transition_cost += transition.cost
            state_id = transition.target
        outcome = -float(transition_cost + states[state_id].terminal_cost)

        for state_index, action in path:
            score = -policy[state_index].copy()
            score[action] += 1.0
            credits = {
                "orm": outcome - evaluation.objective,
                "exact_prm": evaluation.advantage[state_index, action],
            }
            for mode, credit in credits.items():
                estimates[mode][state_index] += probability * credit * score

    for estimate in estimates.values():
        np.testing.assert_allclose(
            estimate,
            evaluation.logit_gradient,
            atol=2e-13,
            rtol=0.0,
        )


def test_synthesis_reports_measured_difficulty():
    constraints = DifficultyConfig(
        min_delayed_choice_rate=0.1,
        max_delayed_choice_rate=0.9,
        min_uniform_regret_per_step=0.1,
        max_uniform_regret_per_step=10.0,
        min_mean_q_gap=1.0,
        max_mean_q_gap=20.0,
        min_mean_depth_entropy=0.5,
        max_generation_attempts=32,
    )
    mdp, report, attempts = synthesize_mdp(
        small_config(), constraints, seed=53, mdp_id="synthesized"
    )
    assert attempts >= 1
    assert report == measure_difficulty(mdp)
    assert report.min_q_gap >= 1.0
    assert max(report.optimal_action_counts) - min(report.optimal_action_counts) <= 1
    assert mdp.metadata["difficulty"] == report.to_dict()
