from functools import partial
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import haiku as hk
from jax.scipy.optimize import minimize

from DEQModel.model.transformer_block import TransformerBlock


class DEQTransformer(hk.Module):
    def __init__(self,
                 num_heads: int,
                 dropout_rate: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    def __call__(self,
                 h: jnp.ndarray,
                 mask: Optional[jnp.ndarray],
                 is_training: bool) -> jnp.ndarray:

        transformer = TransformerBlock(self._num_heads, self._dropout_rate, name='TransformerBlock')
        h = transformer(h, mask, is_training)

        # apply root find on the function and output of transformer
        def g_to_optimise(x):
            return transformer(x, mask, is_training) - x

        h_star = minimize(g_to_optimise, h, method='BFGS')

        return h_star

