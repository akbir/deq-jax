import haiku as hk

import jax.numpy as jnp
from src.model.model import TransformerBlock, TransformerXLBlock


@hk.testing.transform_and_run
def test_transformer_block():
    mod = TransformerBlock(3, 5, 0.1)
    out = mod(jnp.ones([1, 2, 3]), mask=None, is_training=False)
    assert out.ndim == 3



@hk.testing.transform_and_run
def test_xl_transformer_block_with_no_subsequence():
    batch_size, sequence_length, hidden = 7, 75, 120
    num_heads, num_layers, dropout = 12, 2, 0.

    mod = TransformerXLBlock(num_heads, num_layers, dropout)

    # h torch.Size([7, 120, 75])
    # x torch.Size([7, 360, 75])
    # pos_emb torch.Size([1, 120, 75])
    # z0 torch.Size([7, 120, 0])

    h = x = jnp.ones((batch_size, sequence_length, hidden))

    mem = jnp.zeros((num_layers, batch_size, sequence_length, hidden))
    pos = jnp.ones((1, sequence_length, hidden))

    out = mod(h, x, pos, mem, mask=None, is_training=False)
    assert out.ndim == 3



