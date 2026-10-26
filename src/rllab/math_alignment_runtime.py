"""Native worker that captures gradients without changing the actor."""

import math
import torch
from verl.single_controller.base.decorator import Dispatch, register
from rllab.math_alignment_gradients import FrozenGradientBank
from rllab.real_rl_runtime import AuditedActorWorker


class AlignmentWorker(AuditedActorWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_capture(self):
        engine = self.actor.engine
        assert getattr(engine, "scaler", None) is None
        assert math.isinf(engine.optimizer_config.clip_grad)
        assert all(p.dtype == torch.float32 for p in engine.module.parameters())
        self._gradients = FrozenGradientBank(
            engine.module, engine.optimizer, torch.distributed.get_world_size()
        )
        return dict(
            rank=self.rank, weighted_parameter_numel=self._gradients.weighted_numel
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def arm_capture(self, save=None, accumulate=(), references=()):
        self._gradients.arm(save=save, accumulate=accumulate, references=references)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_capture(self):
        assert not self.actor.engine.optimizer.state
        return dict(rank=self.rank, **self._gradients.finish())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def make_directions(self, norms, delta):
        return dict(rank=self.rank, **self._gradients.make_directions(norms, delta))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def combine_gradients(self, destination, coefficients):
        return dict(
            rank=self.rank, **self._gradients.combine(destination, coefficients)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def signal_totals(self, a, b):
        bank = self._gradients
        return dict(
            rank=self.rank,
            local_aa=bank.dot(a, a),
            local_bb=bank.dot(b, b),
            local_ab=bank.dot(a, b),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def drop_gradients(self, names):
        self._gradients.drop(names)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_capture(self):
        bank = self._gradients
        assert bank.pending is None and bank.result is None
        assert all(p.grad is None for p in bank.parameters)
        result = dict(rank=self.rank, intercepted_steps=bank.intercepted_steps)
        bank.close()
        del self._gradients
        return result
