"""Auditable full-parameter, unclipped, no-momentum policy-gradient updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from rllab.experiment.calibration_runtime import CalibrationTable
from rllab.experiment.corpus import ExperimentCorpus
from rllab.experiment.modeling import no_sync_context, unwrap_policy
from rllab.experiment.rollout import RolloutBatch
from rllab.experiment.runtime import DistributedContext


GRADIENT_SYNCHRONIZATION = "per-parameter-all-reduce-rank0-average-broadcast"


@dataclass(frozen=True)
class BackwardMetrics:
    local_loss: float
    global_loss: float
    gradient_l2: float
    examples: int
    episodes: int


@dataclass(frozen=True)
class ExactGradientMetrics:
    loss: float
    gradient_l2: float


def exact_policy_logit_coefficients(
    corpus, snapshot
) -> dict[tuple[str, str], tuple[float, ...]]:
    """Return exact ``d_pi * pi * A_pi`` coefficients at any policy boundary."""

    from rllab.mdp.solve import evaluate_policy, mdp_arrays

    expected_mdps = {bundle.mdp.mdp_id for bundle in corpus.bundles}
    if set(snapshot.policies) != expected_mdps:
        raise ValueError("policy snapshot and experiment corpus contain different MDPs")
    result: dict[tuple[str, str], tuple[float, ...]] = {}
    for bundle in corpus.bundles:
        mdp_id = bundle.mdp.mdp_id
        arrays = mdp_arrays(bundle.mdp)
        evaluation = evaluate_policy(bundle.mdp, snapshot.policy(mdp_id))
        for state_id in bundle.nonterminal_state_ids:
            values = tuple(
                float(value)
                for value in evaluation.logit_gradient[arrays.state_index[state_id]]
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"non-finite exact coefficient for {mdp_id}:{state_id}"
                )
            if abs(sum(values)) > 1.0e-10:
                raise ValueError(
                    f"exact logit coefficients do not sum to zero for {mdp_id}:{state_id}"
                )
            result[(mdp_id, state_id)] = values
    if len(result) != corpus.example_count:
        raise ValueError("exact coefficient table does not cover every prompt")
    return result


def build_optimizer(policy, *, learning_rate: float):
    import torch

    module = unwrap_policy(policy)
    return torch.optim.SGD(
        module.parameters(),
        lr=learning_rate,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        maximize=False,
        foreach=True,
    )


def policy_gradient_backward(
    policy,
    rollout: RolloutBatch,
    calibration: CalibrationTable,
    *,
    optimizer,
    microbatch_states: int,
    context: DistributedContext,
) -> BackwardMetrics:
    import torch

    optimizer.zero_grad(set_to_none=True)
    examples = rollout.examples
    chunks = [
        examples[start : start + microbatch_states]
        for start in range(0, len(examples), microbatch_states)
    ]
    local_loss = torch.zeros((), device="cuda", dtype=torch.float32)
    for chunk_index, chunk in enumerate(chunks):
        with no_sync_context(policy, enabled=False):
            input_ids = torch.tensor(
                [example.prompt.input_ids for example in chunk],
                dtype=torch.long,
                device="cuda",
            )
            bias = torch.tensor(
                [
                    calibration.rows[example.prompt.example_id].fixed_bias
                    for example in chunk
                ],
                dtype=torch.float32,
                device="cuda",
            )
            actions = torch.tensor(
                [example.action for example in chunk], dtype=torch.long, device="cuda"
            )
            credit = torch.tensor(
                [example.credit for example in chunk],
                dtype=torch.float32,
                device="cuda",
            )
            log_probs = policy(input_ids)
            log_probs = torch.log_softmax(log_probs + bias, dim=-1)
            selected = log_probs.gather(1, actions[:, None]).squeeze(1)
            # Divide by episodes, not states: this is E[R * sum_t log pi_t].
            loss = -(credit * selected).sum() / rollout.episodes
            loss.backward()
            local_loss += loss.detach()

    module = unwrap_policy(policy)
    # DDP no_sync accumulation briefly needs both standalone FP32 gradients and
    # ~30 GiB of reducer buckets for this 8B model. Instead, synchronize every
    # completed gradient tensor in place. This has the same averaged gradient
    # and communication volume without a second full-model buffer.
    synchronize_gradients(policy, context=context)
    gradient_l2 = gradient_l2_norm(module)

    global_loss = local_loss.detach().clone()
    if context.distributed:
        torch.distributed.all_reduce(global_loss, op=torch.distributed.ReduceOp.SUM)
        global_loss /= context.world_size
    return BackwardMetrics(
        local_loss=float(local_loss.item()),
        global_loss=float(global_loss.item()),
        gradient_l2=gradient_l2,
        examples=len(examples),
        episodes=rollout.episodes,
    )


def exact_uniform_policy_gradient_backward(
    policy,
    corpus: ExperimentCorpus,
    calibration: CalibrationTable,
    *,
    context: DistributedContext,
    microbatch_states: int,
) -> ExactGradientMetrics:
    """Backpropagate the exact uniform-policy root objective over all states.

    The action coefficients are detached exact
    ``occupancy * probability * advantage`` values from the dataset truth. The
    distributed scale makes the final averaged gradient an equal mean over
    MDPs, independent of world size.
    """

    import torch

    module = unwrap_policy(policy)
    module.train()
    for parameter in module.parameters():
        parameter.grad = None
    prompts = corpus.all_prompts()
    local = tuple(
        prompt
        for index, prompt in enumerate(prompts)
        if index % context.world_size == context.rank
    )
    bundle_by_mdp = corpus.bundle_by_mdp()
    local_loss = torch.zeros((), device="cuda", dtype=torch.float32)
    scale = context.world_size / len(corpus.bundles)
    for start in range(0, len(local), microbatch_states):
        batch = local[start : start + microbatch_states]
        input_ids = torch.tensor(
            [prompt.input_ids for prompt in batch], dtype=torch.long, device="cuda"
        )
        bias = torch.tensor(
            [calibration.rows[prompt.example_id].fixed_bias for prompt in batch],
            dtype=torch.float32,
            device="cuda",
        )
        coefficients = []
        for prompt in batch:
            truth = bundle_by_mdp[prompt.mdp_id].truth[prompt.state_id]
            advantages = [
                float(action["uniform_advantage"])
                for action in sorted(
                    truth["actions"], key=lambda item: int(item["action"])
                )
            ]
            coefficients.append(
                [
                    float(truth["root_uniform_occupancy"])
                    * (1.0 / len(advantages))
                    * advantage
                    for advantage in advantages
                ]
            )
        coefficient = torch.tensor(coefficients, dtype=torch.float32, device="cuda")
        log_probs = torch.log_softmax(policy(input_ids) + bias, dim=-1)
        loss = -(coefficient * log_probs).sum() * scale
        loss.backward()
        local_loss += loss.detach()
    synchronize_gradients(policy, context=context)
    global_loss = local_loss.clone()
    if context.distributed:
        torch.distributed.all_reduce(global_loss, op=torch.distributed.ReduceOp.SUM)
        global_loss /= context.world_size
    return ExactGradientMetrics(
        loss=float(global_loss.item()),
        gradient_l2=gradient_l2_norm(module),
    )


def exact_policy_gradient_backward(
    policy,
    corpus: ExperimentCorpus,
    calibration: CalibrationTable,
    snapshot,
    *,
    context: DistributedContext,
    microbatch_states: int,
) -> ExactGradientMetrics:
    """Backpropagate the exact root objective at an arbitrary frozen policy.

    The detached coefficients are evaluated from the exact MDP at the same
    policy snapshot used for model inference and sampled rollouts. This is the
    non-uniform counterpart of :func:`exact_uniform_policy_gradient_backward`.
    """

    import torch

    module = unwrap_policy(policy)
    module.train()
    for parameter in module.parameters():
        parameter.grad = None
    coefficients = exact_policy_logit_coefficients(corpus, snapshot)
    prompts = corpus.all_prompts()
    local = tuple(
        prompt
        for index, prompt in enumerate(prompts)
        if index % context.world_size == context.rank
    )
    local_loss = torch.zeros((), device="cuda", dtype=torch.float32)
    scale = context.world_size / len(corpus.bundles)
    for start in range(0, len(local), microbatch_states):
        batch = local[start : start + microbatch_states]
        input_ids = torch.tensor(
            [prompt.input_ids for prompt in batch], dtype=torch.long, device="cuda"
        )
        bias = torch.tensor(
            [calibration.rows[prompt.example_id].fixed_bias for prompt in batch],
            dtype=torch.float32,
            device="cuda",
        )
        coefficient = torch.tensor(
            [coefficients[(prompt.mdp_id, prompt.state_id)] for prompt in batch],
            dtype=torch.float32,
            device="cuda",
        )
        log_probs = torch.log_softmax(policy(input_ids) + bias, dim=-1)
        loss = -(coefficient * log_probs).sum() * scale
        loss.backward()
        local_loss += loss.detach()
    synchronize_gradients(policy, context=context)
    global_loss = local_loss.clone()
    if context.distributed:
        torch.distributed.all_reduce(global_loss, op=torch.distributed.ReduceOp.SUM)
        global_loss /= context.world_size
    return ExactGradientMetrics(
        loss=float(global_loss.item()),
        gradient_l2=gradient_l2_norm(module),
    )


def gradient_l2_norm(policy_or_module) -> float:
    """Return the FP64-accumulated L2 norm of the current FP32 gradients."""

    import torch

    module = unwrap_policy(policy_or_module)
    squared_norm = torch.zeros((), device="cuda", dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter_norm = torch.linalg.vector_norm(parameter.grad.detach().float())
            squared_norm += parameter_norm.double().square()
    return float(torch.sqrt(squared_norm).item())


def resolve_step_size(
    *,
    scale_rule: str,
    learning_rate: float,
    target_parameter_step_l2: float | None,
    gradient_l2: float,
) -> tuple[float, float]:
    """Resolve the applied LR and nominal parameter-step L2 for one gradient."""

    if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
        raise ValueError("gradient_l2 must be finite and positive")
    if scale_rule == "fixed_learning_rate":
        step_size = float(learning_rate)
    elif scale_rule == "fixed_parameter_step":
        if target_parameter_step_l2 is None:
            raise ValueError("fixed_parameter_step requires a target L2")
        step_size = float(target_parameter_step_l2) / gradient_l2
    else:
        raise ValueError(f"unknown scale rule: {scale_rule}")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("resolved step size must be finite and positive")
    return step_size, step_size * gradient_l2


def synchronize_gradients(policy, *, context: DistributedContext) -> None:
    """Average gradients and broadcast rank 0's exact FP32 result to every rank."""

    import torch

    if not context.distributed:
        return
    module = unwrap_policy(policy)
    for parameter in module.parameters():
        if parameter.grad is not None:
            torch.distributed.all_reduce(
                parameter.grad, op=torch.distributed.ReduceOp.SUM
            )
            # The vendor collective can return last-bit-different sums on
            # different ranks. Let rank 0 perform the division once, then make
            # that exact tensor authoritative so a sharded evaluation cannot
            # become a mosaic of slightly different policies after repeated
            # fixed-norm updates.
            if context.rank == 0:
                parameter.grad.div_(context.world_size)
            torch.distributed.broadcast(parameter.grad, src=0)


def apply_gradient_fraction(policy, *, step_size: float) -> None:
    """Apply ``theta <- theta - step_size * grad`` without clearing gradients."""

    import torch

    if not math.isfinite(step_size) or step_size < 0:
        raise ValueError("step_size must be finite and non-negative")
    module = unwrap_policy(policy)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-step_size)


def optimizer_contract(optimizer) -> dict[str, Any]:
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("RLLab expects exactly one SGD parameter group")
    group = optimizer.param_groups[0]
    contract = {
        "class": type(optimizer).__name__,
        "learning_rate": float(group["lr"]),
        "momentum": float(group["momentum"]),
        "dampening": float(group["dampening"]),
        "weight_decay": float(group["weight_decay"]),
        "nesterov": bool(group["nesterov"]),
        "maximize": bool(group["maximize"]),
        "gradient_clipping": None,
        "kl_penalty": None,
        "ppo_clipping": None,
        "advantage_normalization": None,
    }
    if contract["class"] != "SGD" or any(
        contract[key] != expected
        for key, expected in (
            ("momentum", 0.0),
            ("dampening", 0.0),
            ("weight_decay", 0.0),
            ("nesterov", False),
            ("maximize", False),
        )
    ):
        raise RuntimeError(f"optimizer violates the experimental contract: {contract}")
    return contract
