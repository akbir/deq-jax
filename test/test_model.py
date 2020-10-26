import functools

import haiku as hk
import jax

import jax.numpy as jnp
import optax
import pytest

from src.model.model import UTBlock
from src.model.train import build_forward_fn, lm_loss_fn, Updater


@hk.testing.transform_and_run
def test_transformer_block():
    mod = UTBlock(3, 5, 0.1)
    input = jnp.ones([1, 2, 3])
    hidden = jnp.ones([1, 2, 3])
    out = mod(input, hidden, mask=None, is_training=False)
    assert out.ndim == 3


@pytest.fixture
def train_dataset():
    batch_size, sequence_length, vocab_size = 16, 128, 100
    rng = jax.random.PRNGKey(0)
    data = jax.random.randint(rng, (batch_size, sequence_length), 0, vocab_size)
    return (data for data in [{'obs': data[:, :-1], 'target': data[:, 1:]}] * 200)


def test_train(train_dataset):
    vocab_size, d_model, num_heads, num_layers = 100, 256, 4, 1
    dropout_rate, grad_clip_value, learning_rate = 0.01, 0.25, 2e-4
    max_iter = 100

    # Set up the model, loss, and updater.
    forward_fn = build_forward_fn(vocab_size, d_model, num_heads,
                                  num_layers, dropout_rate, max_iter)

    forward_fn = hk.transform(forward_fn)
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(learning_rate, b1=0.9, b2=0.99))

    updater = Updater(forward_fn.init, loss_fn, optimizer)

    # Initialize parameters.
    rng = jax.random.PRNGKey(428)
    data = next(train_dataset)
    state = updater.init(rng, data)

    for step in range(10):
        data = next(train_dataset)
        state, metrics = updater.update(state, data)
        print(metrics['loss'])

    assert metrics == None

