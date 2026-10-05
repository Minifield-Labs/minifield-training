"""Quantization warm-up: fit the quantized encoder to the dense one first.

Before the curriculum, the encoder's projections train under the chosen
quantizer (fake-quantized, straight-through) to reproduce the frozen dense
encoder on generic English text: its masked-LM predictions at masked tokens
(KL over the model's vocabulary; the head is the tied, frozen embedding
table) and its final hidden states (cosine distance). Text is read through
the model's vocabulary, one document chunk per row. The curriculum then
starts from these masters, so its updates go to the task rather than to
surviving quantization.
"""

from collections.abc import Callable, Iterator
import dataclasses
import functools
import hashlib
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from safetensors.numpy import save_file

from examples.magicbox import composition as magicbox
from examples.toolcalls import vocabulary
from minifield_training.core import json_io
from minifield_training.kernels import types
from minifield_training.models import contracts
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.strategies import pretrained

EMBEDDINGS = "lfm2.embed_tokens.weight"
type Report = Callable[[dict[str, object]], None]
type Step = Callable[
    ...,
    tuple[types.Parameters, optax.OptState, jax.Array, dict[str, jax.Array]],
]


@dataclasses.dataclass(frozen=True)
class Settings:  # pylint: disable=too-many-instance-attributes
    """One warm-up: data, encoder, quantizer, budget and optimizer."""

    model_dir: Path
    dataset_tokenizer: Path
    text: Path
    output: Path
    encoder_source: contracts.PretrainedSource = encoder.SOURCE
    quantizer: str = "ternary"
    tokens: int = 130_000_000
    rows: int = 16
    sequence_tokens: int = 512
    mask_share: float = 0.25
    learning_rate: float = 1e-4
    warmup_share: float = 0.05
    hidden_weight: float = 1.0
    temperature: float = 1.0
    seed: int = 20261005
    checkpoint_every: int = 500
    log_every: int = 50
    attention: encoder.Attention = encoder.DEFAULT_ATTENTION

    @property
    def steps(self) -> int:
        """Updates needed to read ``tokens`` row tokens."""
        return max(1, self.tokens // (self.rows * self.sequence_tokens))

    def identity(self) -> dict[str, object]:
        """Everything that changes the result; resume requires a match."""
        return {
            "source": dataclasses.asdict(self.encoder_source),
            "vocabulary": json_io.digest_file(vocabulary.PATH),
            "text": self.text.name,
            "quantizer": magicbox.QUANTIZERS[self.quantizer],
            "attention": dataclasses.asdict(self.attention),
            **{
                field: getattr(self, field)
                for field in (
                    "tokens",
                    "rows",
                    "sequence_tokens",
                    "mask_share",
                    "learning_rate",
                    "warmup_share",
                    "hidden_weight",
                    "temperature",
                    "seed",
                )
            },
        }


def documents(path: Path) -> Iterator[str]:
    """Texts in file order, one row group at a time."""
    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        yield from parquet.read_row_group(group, columns=["text"]).column(
            0
        ).to_pylist()


def chunk_rows(
    texts: Iterator[str],
    vocab: vocabulary.Encoder,
    rows: int,
    length: int,
    batch: int = 1024,
) -> npt.NDArray[np.uint16]:
    """``rows`` rows of BOS plus up to ``length - 1`` tokens of one document.

    IDs are the encoder's originals; 0 pads. A document longer than a row
    continues in the next; pieces under ``min(32, length // 4)`` tokens
    are dropped.
    """
    original = np.asarray(vocab.kept, np.uint16)
    out = np.zeros((rows, length), np.uint16)
    filled, shortest = 0, min(32, length // 4)
    pending: list[str] = []

    def flush() -> None:
        nonlocal filled
        for encoded in vocab.tokenizer.encode_batch(
            pending, add_special_tokens=False
        ):
            ids = original[np.asarray(encoded.ids, np.int64)]
            for start in range(0, len(ids), length - 1):
                piece = ids[start : start + length - 1]
                if len(piece) < shortest or filled == rows:
                    continue
                out[filled, 0] = 1
                out[filled, 1 : len(piece) + 1] = piece
                filled += 1
        pending.clear()

    for text in texts:
        pending.append(text)
        if len(pending) == batch:
            flush()
            if filled == rows:
                return out
    flush()
    if filled < rows:
        raise ValueError(f"Text ran out after {filled} of {rows} rows")
    return out


def masked(
    ids: npt.NDArray[np.uint16], step: int, settings: Settings, mask_id: int
) -> tuple[
    npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float32]
]:
    """Masked IDs, the attention mask, and which tokens were masked."""
    real = ids > 0
    rng = np.random.default_rng([settings.seed, step])
    chosen = real & (rng.random(ids.shape) < settings.mask_share)
    chosen[:, 0] = False  # BOS stays.
    return (
        np.where(chosen, mask_id, ids).astype(np.int32),
        real.astype(np.int32),
        chosen.astype(np.float32),
    )


def losses(  # pylint: disable=too-many-arguments,too-many-locals
    trainable: types.Parameters,
    frozen: types.Parameters,
    teacher_hidden: jax.Array,
    head: jax.Array,
    ids: jax.Array,
    mask: jax.Array,
    chosen: jax.Array,
    *,
    cfg: lfm.Config,
    effective: Callable[[types.Parameters], types.Parameters],
    settings: Settings,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """KL to the teacher at masked tokens plus hidden-state cosine distance."""
    hidden = encoder.encode(
        effective({**frozen, **trainable}),
        cfg,
        ids,
        mask,
        bf16=True,
        attention_options=settings.attention,
    ).astype(jnp.float32)
    teacher = teacher_hidden.astype(jnp.float32)
    temperature = settings.temperature
    student_logp = jax.nn.log_softmax(hidden @ head.T / temperature)
    teacher_logp = jax.nn.log_softmax(teacher @ head.T / temperature)
    per_token = jnp.sum(
        jnp.exp(teacher_logp) * (teacher_logp - student_logp), axis=-1
    )
    masked_count = jnp.maximum(chosen.sum(), 1.0)
    kl = (per_token * chosen).sum() / masked_count * temperature**2
    real = mask.astype(jnp.float32)
    # Padding is all zeros; a norm with an epsilon keeps its gradient finite.
    cosine = jnp.sum(hidden * teacher, -1) / jnp.sqrt(
        jnp.sum(hidden**2, -1) * jnp.sum(teacher**2, -1) + 1e-12
    )
    distance = ((1.0 - cosine) * real).sum() / jnp.maximum(real.sum(), 1.0)
    agree = (
        (student_logp.argmax(-1) == teacher_logp.argmax(-1)) * chosen
    ).sum() / masked_count
    total = kl + settings.hidden_weight * distance
    return total, {"kl": kl, "hidden": distance, "agree": agree}


def make_step(
    settings: Settings, cfg: lfm.Config
) -> tuple[Step, optax.GradientTransformation]:
    """The jitted update and its optimizer."""
    plan = magicbox.quantization_plan(cfg, settings.quantizer)

    def effective(params: types.Parameters) -> types.Parameters:
        return {
            name: plan.effective(value) if name in plan.names else value
            for name, value in params.items()
        }

    warm = max(1, int(settings.steps * settings.warmup_share))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(
                0.0,
                settings.learning_rate,
                warm,
                max(settings.steps, warm + 1),
                settings.learning_rate * 0.1,
            ),
            weight_decay=0.0,
        ),
    )
    loss = functools.partial(
        losses, cfg=cfg, effective=effective, settings=settings
    )

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(  # pylint: disable=too-many-arguments
        trainable: types.Parameters,
        state: optax.OptState,
        frozen: types.Parameters,
        teacher: types.Parameters,
        head: jax.Array,
        ids: jax.Array,
        mask: jax.Array,
        chosen: jax.Array,
    ) -> tuple[
        types.Parameters, optax.OptState, jax.Array, dict[str, jax.Array]
    ]:
        teacher_hidden = jax.lax.stop_gradient(
            encoder.encode(
                teacher,
                cfg,
                ids,
                mask,
                bf16=True,
                attention_options=settings.attention,
            )
        )
        (value, parts), grads = jax.value_and_grad(loss, has_aux=True)(
            trainable, frozen, teacher_hidden, head, ids, mask, chosen
        )
        updates, state = optimizer.update(grads, state, trainable)
        return optax.apply_updates(trainable, updates), state, value, parts

    return step, optimizer


def _save_state(path: Path, payload: dict[str, object]) -> None:
    staged = path.with_suffix(".tmp")
    with staged.open("wb") as handle:
        pickle.dump(jax.device_get(payload), handle)
    staged.rename(path)


def run(  # pylint: disable=too-many-locals
    settings: Settings,
    *,
    session_seconds: float,
    report: Report = lambda event: print(json.dumps(event), flush=True),
) -> Path | None:
    """Train until done or the deadline; return the warmed encoder file.

    Rerunning resumes from the last checkpoint with the same settings.
    Returns None when the deadline stops it first.
    """
    deadline = time.monotonic() + session_seconds
    output = settings.output
    output.mkdir(parents=True, exist_ok=True)
    identity = settings.identity()
    key = hashlib.sha256(json_io.canonical(identity).encode()).hexdigest()
    result = output / f"encoder-{key[:12]}.safetensors"
    if result.exists():
        return result
    cfg, params = pretrained.load_verified(
        settings.model_dir, settings.encoder_source, encoder.Adapter()
    )
    vocab = vocabulary.Encoder(settings.dataset_tokenizer)
    tokens_path = output / f"rows-{key[:12]}.npy"
    if not tokens_path.exists():
        rows = chunk_rows(
            documents(settings.text),
            vocab,
            settings.steps * settings.rows,
            settings.sequence_tokens,
        )
        np.save(tokens_path, rows)
    rows = np.load(tokens_path, mmap_mode="r")
    order = np.random.default_rng(settings.seed).permutation(len(rows))
    teacher = {name: jnp.asarray(value) for name, value in params.items()}
    frozen = {EMBEDDINGS: teacher[EMBEDDINGS]}
    head = teacher[EMBEDDINGS][jnp.asarray(vocab.kept, jnp.int32)]
    step_fn, optimizer = make_step(settings, cfg)
    state_path = output / f"state-{key[:12]}.pkl"
    if state_path.exists():
        with state_path.open("rb") as handle:
            saved = pickle.load(handle)  # Written by this function only.
        trainable = jax.tree.map(jnp.asarray, saved["trainable"])
        opt_state = jax.tree.map(jnp.asarray, saved["optimizer"])
        start = int(saved["step"])
    else:
        trainable = {
            name: jnp.array(value)
            for name, value in teacher.items()
            if name != EMBEDDINGS
        }
        opt_state = optimizer.init(trainable)
        start = 0
    mask_id = vocab.token_id("<|mask|>")
    report(
        {"event": "warmup", "step": start, "steps": settings.steps, **identity}
    )
    began, step = time.monotonic(), start
    while step < settings.steps and time.monotonic() < deadline:
        batch = np.asarray(
            rows[
                np.sort(
                    order[step * settings.rows : (step + 1) * settings.rows]
                )
            ]
        )
        ids, mask, chosen = masked(batch, step, settings, mask_id)
        trainable, opt_state, value, parts = step_fn(
            trainable,
            opt_state,
            frozen,
            teacher,
            head,
            jnp.asarray(ids),
            jnp.asarray(mask),
            jnp.asarray(chosen),
        )
        step += 1
        if step % settings.log_every == 0 or step == settings.steps:
            report(
                {
                    "warmup_step": step,
                    "loss": float(value),
                    **{name: float(part) for name, part in parts.items()},
                    "tokens_per_second": (step - start)
                    * settings.rows
                    * settings.sequence_tokens
                    / (time.monotonic() - began),
                }
            )
        if step % settings.checkpoint_every == 0 and step < settings.steps:
            _save_state(
                state_path,
                {"trainable": trainable, "optimizer": opt_state, "step": step},
            )
    if step < settings.steps:
        _save_state(
            state_path,
            {"trainable": trainable, "optimizer": opt_state, "step": step},
        )
        report({"event": "warmup_stopped", "step": step})
        return None
    weights = {**frozen, **trainable}
    save_file(
        {
            name: np.asarray(value, np.float32)
            for name, value in weights.items()
        },
        str(result),
        metadata={"warmup": json.dumps(identity)},
    )
    state_path.unlink(missing_ok=True)
    report({"event": "warmup_done", "path": str(result)})
    return result
