# Minifield Training

The Minifield Python/JAX training worker, packaged as an independent library
with executable quality gates.

## Start here

Use Python 3.12 and uv 0.11.30 for development. The training package also admits
Python 3.13 notebook kernels. From this repository:

```sh
uv sync --locked
git config --local core.hooksPath .githooks
uv run --no-sync python scripts/check_quality.py
```

Read the [repository rules](AGENTS.md) and
[architecture](docs/architecture.md) before adding implementation.

## Install surfaces

The base wheel is dependency-free: host modules (`core`, `artifacts`,
`datasets`, `execution`, `supervisor`) import without JAX. Numerical consumers
install the pinned JAX extra:

```sh
uv pip install 'minifield-training[numerical]'
```

`kernels`, `layers`, `models`, `objectives` and `optimizers` require that
extra. For offline chat-template tokenization, install the pinned `text`
extra:

```sh
uv pip install 'minifield-training[text]'
```

The base wheel continues to import `datasets` without this extra. The development group resolves the same locked JAX plus NumPy for
independent test oracles, so `uv sync --locked` covers both surfaces locally.

The [Polyomino decision example](examples/polyomino/README.md) runs directly from a
clone and uses the shared classifier to fine-tune the pinned Base model. Its
game rules, serializer and gameplay callback stay outside the library wheel.

## What is enforced

- Explicit dependency directions, model-family isolation, host-safe inspection,
  and rejection of common import-path bypasses.
- Copied named record contracts, substantial exact function duplication,
  Pylint textual similarity, and
  production function/module size limits. See [limits](docs/duplication.md).
- Pyink, Ruff, Google Pylint, strict Mypy, pytest, local documentation links,
  source archive/wheel builds and an isolated wheel installation.

The local commit hook runs the fast structural checks. CI runs the full quality
command. GitHub branch protection must require the `CPU quality and package`
job; local hooks can be bypassed. Repository files don't enforce server settings.

## Package boundaries

The [executable dependency policy](architecture.toml) assigns 15 owners:

```text
src/minifield_training/
  core/          Host-safe metadata and contracts
  kernels/       Shared numerical primitives and array types
  layers/        Shared parameterized layer composition
  models/        Model contracts, families, parameters and caches
  optimizers/    Shared update mathematics and state
  objectives/    Supervision, weighting and loss reduction
  artifacts/     Host-safe manifests, hashes and file integrity
  datasets/      Admission, splits, tokenization and verified caches
  batching/      Packing, masks, cursors and device batches
  checkpoints/   Full training state and resume compatibility
  evaluation/    Evaluation contracts and distinct outcome metrics
  engine/        Shared training lifecycle
  strategies/    Composition of supported capabilities
  execution/     Job wire adaptation and worker entrypoints
  supervisor/    Child processes, transport and result delivery
```

Each directory has an ownership README. Consult those module guides for the
implemented surface. A directory's first module adds real code, package
initializers, tests and dependencies.

Shared kernels and layers give consumers one implementation to call.
Their output, gradient, state, precision and device contracts must be qualified
before an optimization becomes a supported path. Numerical dependencies and
qualification evidence belong to the components that introduce them.

See [contributing](CONTRIBUTING.md) for commands and
[documentation ownership](docs/README.md) for where contracts and evidence live.

## MagicBox training

[The TPU notebook](examples/kaggle_magicbox_lfm350m_tpu_v5e_8.ipynb)
trains the LFM2.5 Encoder 350M MagicBox architecture on the completed
`minifield.magicbox/1.0` dataset. It includes an offline tiny-model check,
a 2-update full-model TPU check, encoder/head training with frozen token embeddings, exact resume,
held-out evaluation, inference export, and reload/prediction. Setup clones
the training repository from GitHub and checks out a pinned commit.

It detects single-host TPU device counts and defaults to a 10-update smoke
run. Set `RUN_MODE = 'full'` for the 3-epoch recipe, with a 32-request logical
batch on 8 devices. Both modes download the pinned `protodotdesign/magicbox-v1`
dataset and save to separate run directories.

See [the architecture and run guide](docs/magicbox.md) and
[the example README](examples/magicbox/README.md).

## Tool-call training

[The tool-call notebook](examples/colab_toolcalls_lfm350m_tpu.ipynb) trains
a model that picks the next tool and fills its arguments, on MagicBox's
pointer architecture with trainable role markers. It runs the 4-stage
`protodotdesign/toolcalls-v1` curriculum in order, warm-starting each stage
from the last. See [the example README](examples/toolcalls/README.md).
