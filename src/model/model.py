"""Transformer model components. Modified from https://github.com/deepmind/dm-haiku/blob/master/examples/transformer/model.py"""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class Attention(hk.Module):
    """A general multi-headed attention module."""

    def __init__(self,
                 num_heads: int,
                 init_scale: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._num_heads = num_heads
        self._init_scale = init_scale

    @hk.transparent
    def _multihead_linear(self,
                          inputs: jnp.ndarray,
                          head_dim: int) -> jnp.ndarray:
        """Runs a multi-headed linear over inputs, using the given per-head size."""
        batch_size, sequence_length = inputs.shape[:2]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        out = hk.Linear(head_dim * self._num_heads, w_init=initializer)(inputs)
        shape = (batch_size, sequence_length, self._num_heads, head_dim)
        return jnp.reshape(out, shape)

    def __call__(self,
                 x: jnp.ndarray,
                 y: jnp.ndarray,
                 mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        """Multihead attention over y with queries from x.
        Args:
            x: Shape [B, q_timesteps, H1].
            y: Shape [B, kv_timesteps, H2].
            u: Shape [B, kv_timesteps, H1].
            mask: Attention mask to apply. [{1,B}, 1, {1,q_timesteps}, kv_timesteps].
        Returns:
            Output of the attention with shape [batch, timesteps, hiddens]
            """
        batch_size, q_time, embedding_size = x.shape
        head_dim = embedding_size // self._num_heads
        q = self._multihead_linear(x, head_dim)
        k = self._multihead_linear(y, head_dim)
        v = self._multihead_linear(y, head_dim)

        # Compute attention matrix.
        scale = 1. / np.sqrt(head_dim)
        attention = scale * jnp.einsum('bthd,bThd->bhtT', q, k)
        if mask is not None:
            attention = attention * mask - 1e10 * (1 - mask)
        attention = jax.nn.softmax(attention)

        # Attend over values, flatten, and return linear result.
        attended_v = jnp.einsum('bhtT,bThd->bthd', attention, v)
        attended_v = jnp.reshape(attended_v, [batch_size, q_time, embedding_size])
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        return hk.Linear(embedding_size, w_init=initializer)(attended_v)

class CausalSelfAttention(Attention):
    """Self attention with a causal mask applied."""

    def __call__(self, h: jnp.ndarray, mask: Optional[jnp.ndarray], **kwargs) -> jnp.ndarray:
        seq_len = h.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if mask is not None:
            mask *= causal_mask
        else:
            mask = causal_mask
        return super().__call__(h, h, mask)


class DenseBlock(hk.Module):
    """A 2-layer MLP which widens then narrows the input."""

    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)


class UTBlock(hk.Module):
    """A universal transformer block."""

    def __init__(self,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 name: Optional[str] = None):

        super().__init__(name=name)
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate

    def __call__(self,
                 h: jnp.ndarray,
                 input_embs: jnp.ndarray,
                 mask: Optional[jnp.ndarray],
                 is_training: bool) -> jnp.ndarray:
        """Connects the transformer.
        Args:
          input_embs: Inputs, [B, T, H].
          h: Hidden, [B, T, H].
          h: Hidden, [B, T, H].
          mask: Padding mask, [B, T].
          is_training: Whether we're training or not.
        Returns:
          Array of shape [B, T, H].
        """

        init_scale = 2. / np.sqrt(self._num_layers)
        dropout_rate = self._dropout_rate if is_training else 0.
        if mask is not None:
            mask = mask[:, None, None, :]

        for i in range(self._num_layers):
            # input injections
            h = h + input_embs

            # regular transformer block
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            h_attn = CausalSelfAttention(self._num_heads,
                                         init_scale,
                                         name=f'h{i}_attn')(h_norm, mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense
        h = layer_norm(h, name='ln_f')

        return h

def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)
