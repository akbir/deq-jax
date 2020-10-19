"""Transformer model components."""
from typing import Optional, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

__author__ = "akbir"

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


class XLAttention(hk.Module):
    """Multi-headed attention for the transformer XL."""

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
        out = hk.Linear(head_dim * self._num_heads,
                        w_init=initializer,
                        with_bias=False)(inputs)
        shape = (batch_size, sequence_length, self._num_heads, head_dim)
        return jnp.reshape(out, shape)

    @hk.transparent
    def _relative_pos(self,
                      inputs: jnp.ndarray,
                      head_dim: int) -> jnp.ndarray:
        """Calculates the relative position."""
        batch_size, sequence_length = inputs.shape[:2]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        out = hk.Linear(head_dim * self._num_heads,
                        w_init=initializer,
                        with_bias=False)(inputs)
        shape = (batch_size, sequence_length, self._num_heads, head_dim)
        return jnp.reshape(out, shape)

    def _rel_shift(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculates the relative position."""
        return x

    def __call__(self,
                 h: jnp.ndarray,
                 pos: jnp.ndarray,
                 x: jnp.ndarray,
                 mem: jnp.ndarray,
                 mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        """Relative positional attention over x.
        Args:
            h: Shape [B, q_timesteps, H1].
            pos: Shape [B, kv_timesteps, H2].
            x: Shape [B, q_timesteps, H1].
            mem: Shape [B, q_timesteps, H1].
            mask: Attention mask to apply. [{1,B}, 1, {1,q_timesteps}, kv_timesteps].
        Returns:
            Output of the attention with shape [batch, timesteps, hiddens]
            """

        h_with_mem = jnp.concatenate((jax.lax.stop_gradient(mem), h), axis=1)

        batch_size, q_len, embedding_size = h.shape
        r_len = pos.shape[1]
        k_len = h_with_mem.shape[1]

        head_dim = embedding_size // self._num_heads

        initializer = hk.initializers.VarianceScaling(self._init_scale)

        content_bias = hk.get_parameter("c_bias",
                                        shape=[self._num_heads, head_dim],
                                        init=initializer)

        positional_bias = hk.get_parameter("p_bias",
                                           shape=[self._num_heads, head_dim],
                                           init=initializer)

        # calculate attention weights
        r = self._relative_pos(pos, head_dim)
        q = self._multihead_linear(h, head_dim)
        k = self._multihead_linear(h_with_mem, head_dim)
        v = self._multihead_linear(h_with_mem, head_dim)

        # input injection
        x_q = jnp.reshape(x, (batch_size, q_len, self._num_heads, head_dim))
        x_kv = jnp.reshape(x, (batch_size, k_len, self._num_heads, head_dim))

        q += x_q
        k += x_kv
        v += x_kv

        # Todo: set v = v[:,:,-qlen:]

        # Compute attention matrix.
        AC = jnp.einsum('bthd,bThd->bhtT', q + content_bias, k)
        BD = self._rel_shift(jnp.einsum('bthd,Thd->bhtT', q+positional_bias, r))
        scale = 1. / np.sqrt(head_dim)
        attention = scale * (AC + BD)

        if mask is not None:
            attention = attention * mask - 1e10 * (1 - mask)

        attention = jax.nn.softmax(attention)

        # Attend over values, flatten, and return linear result.
        attended_v = jnp.einsum('bhtT,bThd->bthd', attention, v)
        attended_v = jnp.reshape(attended_v, [batch_size, q_len, embedding_size])
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        return hk.Linear(embedding_size, w_init=initializer)(attended_v)


class CausalSelfAttention(Attention):
    """Self attention with a causal mask applied."""

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray], **kwargs) -> jnp.ndarray:
        seq_len = x.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if mask is not None:
            mask *= causal_mask
        else:
            mask = causal_mask
        return super().__call__(x, x, mask)


class CausalXLAttention(XLAttention):
    """Relative attention with a causal mask applied."""

    def __call__(self,
                 h: jnp.ndarray,
                 pos: jnp.ndarray,
                 x: jnp.ndarray,
                 mem: jnp.ndarray,
                 mask: Optional[jnp.ndarray],
                 **kwargs) -> jnp.ndarray:

        seq_len = x.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if mask is not None:
            mask *= causal_mask
        else:
            mask = causal_mask
        return super().__call__(h, pos, x, mem, mask)


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


class TransformerBlock(hk.Module):
    """A transformer block."""

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
                 mask: Optional[jnp.ndarray],
                 is_training: bool) -> jnp.ndarray:
        """Connects the transformer.
        Args:
          h: Inputs, [B, T, H].
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


class TransformerXLBlock(hk.Module):
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
                 x: jnp.ndarray,
                 pos_emb: jnp.ndarray,
                 mem: jnp.ndarray,
                 mask: Optional[jnp.ndarray],
                 is_training: bool) -> jnp.ndarray:
        """Transformer-XL block
        Args:
          h: Hidden input, [B, T, H].
          x: Original Inputs, [B, T, H].
          pos_emb: Positional Embeddings [B, T]
          mem: Historical hidden [N, B, T, H]
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
            mem = mem[:i]
            h_attn = CausalXLAttention(self._num_heads,
                                       init_scale,
                                       name=f'h{i}_attn')(h, pos_emb, x, mem, mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h_norm = layer_norm(h + h_attn, name='ln_f')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
        h = layer_norm(h_dense, name='ln_f')

        return h


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)
