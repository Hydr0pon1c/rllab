from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rllab.config import GeneratorConfig
from rllab.errors import ConfigurationError
from rllab.experiment.config import experiment_config_from_dict, load_experiment_config
from rllab.experiment.cli import build_parser as build_experiment_parser
from rllab.experiment.corpus import (
    ExperimentCorpus,
    InstanceBundle,
    LabelingInfo,
    PromptExample,
    load_experiment_corpus,
)
from rllab.experiment.evaluation import (
    PolicySnapshot,
    policy_snapshot_fingerprint,
    policy_snapshot_l2_difference,
)
from rllab.experiment.kernel import directional_kernel_metrics
from rllab.experiment.optimization import (
    exact_policy_logit_coefficients,
    resolve_step_size,
)
from rllab.experiment.rollout import (
    assigned_episode_ids,
    collect_rollouts,
    episode_action_uniforms,
    inverse_cdf_actions,
    rollout_trajectory_rows,
    trajectory_fingerprint,
)
from rllab.experiment.train import _sample_budget
from rllab.mdp.generate import generate_layered_mdp
from rllab.mdp.solve import evaluate_policy, mdp_arrays, uniform_policy


def test_experiment_config_is_strict_and_preserves_scientific_contract():
    config = experiment_config_from_dict(
        {
            "training": {
                "credit_mode": "orm",
                "steps": 5,
                "kernel_probe_steps": [0, 4],
            }
        }
    )
    assert config.training.credit_mode == "orm"
    assert config.training.kernel_probe_steps == (0, 4)
    assert config.model.master_dtype == "float32"
    with pytest.raises(ConfigurationError):
        experiment_config_from_dict({"training": {"momentum": 0.9}})
    with pytest.raises(ConfigurationError):
        experiment_config_from_dict({"runtime": {"allow_tf32": True}})
    with pytest.raises(ConfigurationError):
        experiment_config_from_dict(
            {"training": {"steps": 2, "kernel_probe_steps": [1, 0]}}
        )


def test_fixed_step_and_explicit_evaluation_schedule_are_strict():
    config = experiment_config_from_dict(
        {
            "training": {
                "steps": 32,
                "scale_rule": "fixed_parameter_step",
                "target_parameter_step_l2": 4.346703360702091e-6,
                "evaluation_steps": [1, 2, 4, 8, 16, 32],
                "kernel_probe_steps": [0, 3, 15, 31],
            }
        }
    )
    assert config.training.evaluation_steps == (1, 2, 4, 8, 16, 32)
    with pytest.raises(ConfigurationError, match="target_parameter_step_l2"):
        experiment_config_from_dict(
            {"training": {"scale_rule": "fixed_parameter_step"}}
        )
    with pytest.raises(ConfigurationError, match="evaluation_steps"):
        experiment_config_from_dict(
            {"training": {"steps": 4, "evaluation_steps": [1, 5]}}
        )


def test_checked_in_puzzlemdp_experiment_configs_are_valid():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for folder in (root / "configs/experiment",)
        for path in folder.glob("*.yaml")
    )
    assert paths
    for path in paths:
        config = load_experiment_config(path)
        assert config.schema_version == "rllab-experiment-config/v1"


def test_train_cli_accepts_a_scientific_seed_override():
    args = build_experiment_parser().parse_args(
        ["train", "--config", "condition.yaml", "--seed", "20260829"]
    )
    assert args.seed == 20260829


def test_episode_assignment_keeps_b16_balanced_and_nested():
    num_mdps = 8
    for world_size in (2, 8, 16):
        assigned = {
            rank: [
                (mdp_index, episode)
                for mdp_index in range(num_mdps)
                for episode in assigned_episode_ids(
                    mdp_index=mdp_index,
                    num_mdps=num_mdps,
                    episodes_per_mdp=8,
                    rank=rank,
                    world_size=world_size,
                )
            ]
            for rank in range(world_size)
        }
        assert all(len(rows) == 64 // world_size for rows in assigned.values())
        assert all(
            sum(episode < 2 for _, episode in rows) == 16 // world_size
            for rows in assigned.values()
        )
        flattened = sorted(row for rows in assigned.values() for row in rows)
        assert flattened == [
            (mdp_index, episode)
            for mdp_index in range(num_mdps)
            for episode in range(8)
        ]


def test_explicit_inverse_cdf_sampler_has_auditable_boundaries():
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.4]] * 5, dtype=np.float64)
    uniforms = np.asarray([0.0, 0.1, 0.299, 0.301, 0.999], dtype=np.float64)
    assert inverse_cdf_actions(probabilities, uniforms).tolist() == [0, 1, 1, 2, 3]


def test_evaluated_policy_rollout_pairs_orm_and_prm_without_another_forward():
    mdp = generate_layered_mdp(
        GeneratorConfig(horizon=2, width=4, terminal_width=4, num_actions=4),
        seed=71,
        mdp_id="rollout-test",
    )
    prompts = {}
    for state in mdp.states:
        if state.terminal:
            continue
        prompts[state.state_id] = PromptExample(
            example_id=f"example-{state.state_id}",
            mdp_id=mdp.mdp_id,
            labeling_id="labels",
            state_id=state.state_id,
            depth=state.depth,
            input_ids=(1, 2, 3),
            prompt_sha256="0" * 64,
            action_token_ids=(32, 33, 34, 35),
            current_state_token_position=1,
            state_slots=((1, "current_state", state.state_id),),
        )
    bundle = InstanceBundle(
        mdp=mdp,
        split="train",
        labeling=LabelingInfo(
            labeling_id="labels",
            mdp_id=mdp.mdp_id,
            mode="random",
            state_token_ids={state_id: index for index, state_id in enumerate(prompts)},
            action_token_ids=(32, 33, 34, 35),
            source="test",
        ),
        prompts=prompts,
        truth={},
    )
    corpus = ExperimentCorpus(
        identity=None,  # type: ignore[arg-type]
        bundles=(bundle,),
        action_token_ids=(32, 33, 34, 35),
        prompt_length=3,
        labeling_mode="random",
        split="train",
    )
    episode_ids = (0, 2)
    uniforms = episode_action_uniforms(
        seed=20260825,
        update=0,
        mdp_index=0,
        episode_ids=episode_ids,
        horizon=mdp.horizon,
    )
    policy = uniform_policy(mdp)
    evaluation = evaluate_policy(mdp, policy)
    coefficients = exact_policy_logit_coefficients(
        corpus, PolicySnapshot({mdp.mdp_id: policy})
    )
    arrays = mdp_arrays(mdp)
    for state_id in bundle.nonterminal_state_ids:
        assert coefficients[(mdp.mdp_id, state_id)] == pytest.approx(
            evaluation.logit_gradient[arrays.state_index[state_id]]
        )
    orm = collect_rollouts(
        bundle,
        policy,
        credit_mode="orm",
        episode_ids=episode_ids,
        action_uniforms=uniforms,
    )
    prm = collect_rollouts(
        bundle,
        policy,
        credit_mode="exact_prm",
        episode_ids=episode_ids,
        action_uniforms=uniforms,
    )
    assert [example.action for example in orm.examples] == [
        example.action for example in prm.examples
    ]
    assert orm.total_returns == prm.total_returns
    assert all(
        example.action_probability == pytest.approx(0.25) for example in orm.examples
    )
    assert all(
        example.credit
        == pytest.approx(orm.total_returns[example.episode] - evaluation.objective)
        for example in orm.examples
    )
    assert all(
        example.credit
        == pytest.approx(
            evaluation.advantage[
                arrays.state_index[example.prompt.state_id], example.action
            ]
        )
        for example in prm.examples
    )
    orm_rows = rollout_trajectory_rows(orm, uniforms, seed=20260825, update=0)
    prm_rows = rollout_trajectory_rows(prm, uniforms, seed=20260825, update=0)
    assert trajectory_fingerprint(orm_rows) == trajectory_fingerprint(prm_rows)


def test_episode_uniforms_preserve_audit_update_zero_and_key_later_updates():
    actual = episode_action_uniforms(
        seed=11, update=0, mdp_index=3, episode_ids=(2,), horizon=5
    )
    expected = np.random.default_rng(
        np.random.SeedSequence([0x524C4C42, 11, 3, 2])
    ).random(5, dtype=np.float64)
    assert np.array_equal(actual[0], expected)
    later = episode_action_uniforms(
        seed=11, update=1, mdp_index=3, episode_ids=(2,), horizon=5
    )
    assert np.array_equal(
        later,
        episode_action_uniforms(
            seed=11, update=1, mdp_index=3, episode_ids=(2,), horizon=5
        ),
    )
    assert not np.array_equal(actual, later)


def test_fixed_parameter_step_resolves_one_common_update_norm():
    effective_lr, step_l2 = resolve_step_size(
        scale_rule="fixed_parameter_step",
        learning_rate=1.25e-10,
        target_parameter_step_l2=4.0e-6,
        gradient_l2=20_000.0,
    )
    assert effective_lr == pytest.approx(2.0e-10)
    assert step_l2 == pytest.approx(4.0e-6)


def test_policy_snapshot_fingerprint_and_path_distance_are_exact():
    base = PolicySnapshot({"mdp": np.asarray([[0.25, 0.25, 0.25, 0.25]])})
    moved = PolicySnapshot({"mdp": np.asarray([[0.30, 0.20, 0.25, 0.25]])})
    assert policy_snapshot_fingerprint(base) == policy_snapshot_fingerprint(base)
    assert policy_snapshot_fingerprint(base) != policy_snapshot_fingerprint(moved)
    assert policy_snapshot_l2_difference(base, moved) == pytest.approx(
        np.sqrt(0.05**2 + 0.05**2)
    )


def test_experiment_corpus_loads_one_complete_paired_condition():
    corpus = load_experiment_corpus(
        "data/smoke/latest",
        split="train",
        labeling_mode="random",
    )
    assert len(corpus.bundles) == 1
    assert corpus.example_count == 9
    assert corpus.prompt_length == 94
    assert corpus.action_token_ids == (32, 33, 34, 35)
    bundle = corpus.bundles[0]
    assert set(bundle.prompts) == set(bundle.nonterminal_state_ids)
    assert all(
        prompt.input_ids[prompt.current_state_token_position]
        == bundle.labeling.state_token_ids[prompt.state_id]
        for prompt in bundle.prompts.values()
    )


def test_directional_kernel_recovers_an_exact_linear_policy_response():
    mdp = generate_layered_mdp(
        GeneratorConfig(horizon=3, width=4, terminal_width=4, num_actions=4),
        seed=17,
        mdp_id="kernel-test",
    )
    arrays = mdp_arrays(mdp)
    base = uniform_policy(mdp)
    updated = base.copy()
    updated[~arrays.terminal, 0] += 0.01
    updated[~arrays.terminal, 1] -= 0.01
    fraction = 0.2
    fractional = base + fraction * (updated - base)
    state_ids = tuple(arrays.state_ids)
    bundle = InstanceBundle(
        mdp=mdp,
        split="train",
        labeling=LabelingInfo(
            labeling_id="labels",
            mdp_id=mdp.mdp_id,
            mode="random",
            state_token_ids={
                state_id: index for index, state_id in enumerate(state_ids)
            },
            action_token_ids=(32, 33, 34, 35),
            source="test",
            assignment_score=None,
            assignment_metadata={},
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
    config = experiment_config_from_dict({"training": {"trajectories_global": 64}})
    assert _sample_budget(config, corpus, 5) == {
        "horizon": 3,
        "trajectories_per_update": 64,
        "action_tokens_per_update": 192,
        "cumulative_trajectories": 320,
        "cumulative_action_tokens": 960,
    }
    rows, summary = directional_kernel_metrics(
        corpus,
        PolicySnapshot({mdp.mdp_id: base}),
        PolicySnapshot({mdp.mdp_id: fractional}),
        PolicySnapshot({mdp.mdp_id: updated}),
        probe_fraction=fraction,
        policy_loss_states={mdp.mdp_id: [mdp.start_state]},
    )
    assert len(rows) == len(mdp.states)
    assert summary["policy_delta_rmse"] < 1e-15
    assert summary["policy_delta_correlation"] == pytest.approx(1.0)
    assert summary["exposure"]["token_absent"]["states"] > 0
