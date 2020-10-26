# Deep Equilibrium Models [NeurIPS'19]
Jax Implementation for the deep equilibrium (DEQ) model, an implicit-depth architecture proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun.

Unlike many existing "deep" techniques, the DEQ model is a implicit-depth architecture that directly solves for and backpropagates through the equilibrium state of an (effectively) infinitely deep network. 

## Prerequisite
Python >= 3.5 and Haiku >= 0.0.2

## Major Components
This repo provides the following re-usable components:

1. JAX implementation of the Broyden's method, a quasi-Newton method for finding roots in k variables. This method is JIT-able
2. JAX implementation DEQ model (custom backwards method) for Haiku pure functions
3. Haiku implementation of the Transformer with input injections

## Usage
All DEQ instantiations share the same underlying framework, whose core functionalities are provided in src/modules.
In particular, `rootfind.py` provides the Jax functions that solves for the roots in forward and backward passes. `broyden.py` provides an implementation of the Broyden's method.

 ```python
import haiku as hk
import jax
import jax.numpy as jnp
from jax import value_and_grad

from deq_jax.src.modules.deq import deq

def build_forward(output_size, max_iter):
    def forward_fn(x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        # create original layers and transform them 
        network = hk.Linear(output_size, name='l1')
        transformed_net = hk.transform(network)

        # lift params
        inner_params = hk.experimental.lift(
            transformed_net.init)(hk.next_rng_key(), x)
        
        # apply deq to functions of form f(params, rng, z, *args)
        z = deq(inner_params, hk.next_rng_key(), x,
                 transformed_net.apply, max_iter, is_training)

        return hk.Linear(output_size)(z)
    return forward_fn

input = jnp.ones((1, 2, 3))
rng = jax.random.PRNGKey(42)
forward_fn = build_forward(3, 10)
forward_fn = hk.transform(forward_fn)
params = forward_fn.init(rng, input)

@jax.jit
def loss_fn(params, rng, x):
    h = forward_fn.apply(params, rng, x)
    return jnp.sum(h)


value, grad = value_and_grad(loss_fn)(params, rng, jnp.ones((1, 2, 3)))
```

To run the Transformer example,  we write the forward method as:

```python
def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                     num_layers: int, dropout_rate: float, max_iter: int):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        """Forward pass."""
        tokens = data['obs']
        input_mask = jnp.greater(tokens, 0)
        batch_size, seq_length = tokens.shape

        # Embed the input tokens and positions.
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
        token_embedding = token_embedding_map(tokens)
        positional_embeddings = hk.get_parameter(
            'pos_embs', [seq_length, d_model], init=embed_init)

        x = token_embedding + positional_embeddings

        # Create transformer block
        transformer_block = model.UTBlock(
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate)

        transformed_net = hk.transform(transformer_block)

        # Lift params
        inner_params = hk.experimental.lift(
            transformed_net.init)(hk.next_rng_key(), x, x, input_mask, is_training)

        # f(_params, _rng, _zz, *args) 
        def fun(_params, _rng, _z, *args):
            return transformed_net.apply(_params, _rng, _z, *args, is_training=is_training)
        
        # Find z* hidden state at equilbrium point
        z_0 = jnp.zeros_like(x)
        
        z_star = deq(inner_params, hk.next_rng_key(), z_0, fun,
                     max_iter, is_training, x, input_mask)

        # Reverse the embeddings (untied).
        return hk.Linear(vocab_size)(z_star)

    return forward_fn

```


For more details on running the Transformer example look into `train.py`. 
## Credits
The repo takes direct inspiration from the [original implementation](https://github.com/locuslab/deq/tree/master) by Shaojie in Torch.
 The transformer module is modified from an [example](https://github.com/deepmind/dm-haiku/blob/master/examples/transformer/model.py) provided by Haiku.