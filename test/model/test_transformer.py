import functools
from typing import Mapping

import jax

from DEQModel.model.model import TransformerBlock
import haiku as hk
import jax.numpy as jnp


def build_forward_fn(vocab_size: int,
                     d_model: int,
                     num_heads: int,
                     dropout_rate: float):
  """Create the model's forward pass."""

  def forward_fn(data: Mapping[str, jnp.ndarray],
                 is_training: bool = True) -> jnp.ndarray:
    """Forward pass."""
    tokens = data['obs']
    input_mask = jnp.greater(tokens, 0)
    seq_length = tokens.shape[1]

    # Embed the input tokens and positions.
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
    token_embs = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter(
        'pos_embs', [seq_length, d_model], init=embed_init)
    input_embeddings = token_embs + positional_embeddings

    # Run the transformer over the inputs.
    transformer = TransformerBlock(
        num_heads=num_heads, dropout_rate=dropout_rate)
    output_embeddings = transformer(input_embeddings, input_mask, is_training)

    # Reverse the embeddings (untied).
    return hk.Linear(vocab_size)(output_embeddings)
  return forward_fn


def lm_loss_fn(forward_fn,
               vocab_size: int,
               params,
               rng,
               data: Mapping[str, jnp.ndarray],
               is_training: bool = True) -> jnp.ndarray:
  """Compute the loss on data wrt params."""
  logits = forward_fn(params, rng, data, is_training)
  targets = jax.nn.one_hot(data['target'], vocab_size)
  assert logits.shape == targets.shape

  mask = jnp.greater(data['obs'], 0)
  loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
  loss = jnp.sum(loss * mask) / jnp.sum(mask)

  return loss


def test_model_forward_pass():
    vocab_size, d_model, num_heads, num_layers, dropout_rate = 512, 20, 4, 1, 0.5

    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward_fn(vocab_size, d_model, num_heads, dropout_rate)
    forward_fn = hk.transform(forward_fn)

    data = {'obs': jnp.ones((2,5), dtype=jnp.int32), 'target': jnp.ones((2,5))}
    params = forward_fn.init(rng, data)
    output_embeddings = forward_fn.apply(params, rng, data, is_training=True)
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size, is_training=True)
    jit_loss = jax.jit(loss_fn)

    output_embeddings = jit_loss(params, rng, data)
    print(output_embeddings)