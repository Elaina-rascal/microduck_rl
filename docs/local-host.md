# Local host environment

Use `scripts/local-env.sh` to run the official project on this host. It fixes
Python to 3.12 and keeps uv in `.venv-tools/bin`, dependencies in `.venv`, and
download/build caches in `.cache-local`. These directories are ignored by Git.
It does not modify system Python or shell startup files. This is Python dependency
isolation; processes still use the host filesystem and NVIDIA driver.

## Setup

The local uv installation uses the official installer with an unmanaged,
project-local destination. To recreate it from the repository root:

```bash
curl --fail --location https://astral.sh/uv/install.sh --output /tmp/microduck-uv-install.sh
UV_UNMANAGED_INSTALL="$PWD/.venv-tools/bin" sh /tmp/microduck-uv-install.sh
./scripts/local-env.sh sync --locked
```

`--locked` installs the versions already resolved in `uv.lock`. Do not replace
the CUDA PyTorch packages with a CPU wheel. A separate system CUDA toolkit is
not required for the packaged training dependencies; the host NVIDIA driver
must be available to the process.

## Use

No shell activation is needed:

```bash
./scripts/local-env.sh run --locked list-envs
./scripts/local-env.sh run --locked --with pytest pytest tests/
```

Before any long training run, run the mandatory small smoke test. TensorBoard
keeps the smoke run local and does not require a W&B account:

```bash
./scripts/local-env.sh run --locked train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.logger tensorboard --agent.run-name local-smoke \
  --enable-nan-guard True
```

Training logs and checkpoints are under `logs/`. Use `scripts/export.py` for ONNX
export so observation normalization is baked into the graph. In Codex, GPU
commands need access outside the device-restricted execution sandbox.

## Verified on 2026-10-13

- Host: Linux x86_64, RTX 3080 Ti (12 GB), NVIDIA driver 580.178.04.
- Environment: Python 3.12.3, uv 0.13.0, PyTorch 2.9.1+cu128, mjlab 1.3.0,
  Warp 1.12.0, MuJoCo 3.10.0. PyTorch tensor computation and Warp GPU
  initialization both passed; `uv pip check` reported compatible packages.
- Registry: 45 tasks loaded. Upstream `list-envs` returns the task count as
  its exit code (45 here), even when listing succeeds.
- Existing regression suite: 263 passed, 1 skipped (ARM-specific GPU test),
  with two existing ONNX deprecation warnings.
- Walking smoke: 64 environments, 5 iterations, 7,680 transitions on CUDA.
  Actor observations were 61D and actions 14D. All 46 scalar log series were
  finite, NaN terminations stayed at zero, and all penalty terms stayed <= 0.
- Checkpoint: `logs/rsl_rl/velocity/2026-10-13_14-53-53_local-smoke/model_4.pt`.
- Export: `artifacts/local-host/smoke-policy.onnx`, generated through
  `scripts/export.py`. ONNX checker and CPU inference passed with input
  `[1, 61]`, output `[1, 14]`, embedded normalization, and finite outputs.

The smoke checkpoint verifies the toolchain only; five iterations do not
produce a trained walking policy.
