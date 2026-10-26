# RLLab contributor guide

Read README.md and docs/development.md before changing code. This public tree
maintains PuzzleMDP and the final online MC16/32 training method with uniform
and outcome-credit controls.

- Inspect Git status first; preserve unrelated user edits.
- Preserve prompt/token invariants, shared masked softmax, uniform initialization,
  exact evaluation and paired PuzzleMDP conditions.
- Preserve explicit math credit/loss/budget settings, full-parameter momentum-free
  SGD, and the established verl/vLLM stack. Never replace vendor PPU Torch.
- New experiments require a minimal question/intervention/measurement/budget
  contract. Maintenance and existing regression tests do not. Null findings are
  valid; do not tune experiments to make PRM win.
- Use scripts/run_task.sh for formal runs. Default to one node with eight devices
  without changing global scientific controls. Frozen running checkouts and
  plans must not be modified.
- Keep weights, data, checkpoints and logs outside Git. Original results and
  predictions are immutable; reanalysis writes new files. Publish reusable code
  and reproduction inputs; exclude manuscripts and private research notes.
- Run proportional existing CPU checks and the native environment for framework
  checks. Report skipped resources and unverified accelerator behavior.
