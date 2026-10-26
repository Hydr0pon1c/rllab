"""A small deterministic puzzle with exact evaluation and reversible state."""

from rllab.config import GeneratorConfig
from rllab.mdp import PuzzleEnv, evaluate_policy, generate_layered_mdp, uniform_policy
from rllab.mdp.solve import optimal_policy


def main():
    mdp = generate_layered_mdp(
        GeneratorConfig(horizon=3, width=4, terminal_width=4, num_actions=4),
        seed=7,
        mdp_id="example",
    )
    uniform = evaluate_policy(mdp, uniform_policy(mdp))
    optimal = evaluate_policy(mdp, optimal_policy(mdp))
    print(f"Uniform expected return: {uniform.objective:.3f}")
    print(f"Optimal expected return: {optimal.objective:.3f}")

    env = PuzzleEnv(mdp)
    snapshot = env.snapshot()
    alternatives = env.counterfactuals()
    result = env.step(0)
    assert result == alternatives[0]
    env.restore(snapshot)
    assert env.snapshot() == snapshot
    print(f"Restored {env.state_id}; {len(alternatives)} exact counterfactuals.")


if __name__ == "__main__":
    main()
