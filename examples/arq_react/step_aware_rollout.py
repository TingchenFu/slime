"""Step- and epoch-aware rollout wrapper.

Drop-in replacement for ``slime.rollout.sglang_rollout.generate_rollout`` that
injects the *current* rollout_id and dataset epoch_id into ``sample.metadata``
before the sample reaches the custom generate function. This is path A from
the design discussion: the outer rollout function is the only place that
*has* ``rollout_id`` on its signature, so we tag samples here and let
``examples.arq_react.rollout.generate`` read them via metadata.

Inside ``generate(args, sample, sampling_params)`` you can then do::

    rollout_id = sample.metadata.get("current_rollout_id")
    epoch_id   = sample.metadata.get("current_epoch")

Both values are resume-safe: ``rollout_id`` is the trainer's main-loop step
(``train.py``'s ``for rollout_id in range(args.start_rollout_id, args.num_rollout)``);
``epoch_id`` is read from ``RolloutDataSource.epoch_id``, which is restored
from the checkpoint on resume (``slime/rollout/data_source.py``).

Wire via::

    --rollout-function-path examples.arq_react.step_aware_rollout.generate_rollout

When ``--eval-function-path`` is left unset slime aliases it to
``--rollout-function-path``; the ``evaluation=True`` branch below passes
through to the default eval path unchanged (eval samples will not carry
these metadata keys; only train samples do).
"""

import functools

from slime.rollout.sglang_rollout import generate_rollout as _default_generate_rollout


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        return _default_generate_rollout(args, rollout_id, data_source, evaluation=True)

    orig_get_samples = data_source.get_samples

    @functools.wraps(orig_get_samples)
    def get_samples_tagged(num_samples):
        groups = orig_get_samples(num_samples)
        # epoch_id may advance inside orig_get_samples when a call straddles
        # the dataset boundary; read it *after* the call so the tag reflects
        # the epoch the samples actually belong to (last-wins for the
        # straddle case, which is the conventional choice).
        epoch_id = getattr(data_source, "epoch_id", None)
        for group in groups:
            for sample in group:
                if sample.metadata is None:
                    sample.metadata = {}
                sample.metadata["current_rollout_id"] = rollout_id
                if epoch_id is not None:
                    sample.metadata["current_epoch"] = epoch_id
        return groups

    data_source.get_samples = get_samples_tagged
    try:
        return _default_generate_rollout(args, rollout_id, data_source, evaluation=False)
    finally:
        data_source.get_samples = orig_get_samples
