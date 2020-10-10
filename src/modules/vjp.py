import collections
import functools
from typing import Any, Mapping, MutableMapping, Tuple, TypeVar, Callable, Union

import jax
import jax.numpy as jnp
from haiku._src import base
from haiku._src.stateful import difference

InternalState = collections.namedtuple("InternalState", "params,state,rng")
Bundle = Mapping[str, Mapping[str, Any]]
T = TypeVar("T")

def copy_structure(bundle: T) -> T:
  return jax.tree_map(lambda x: x, bundle)


def internal_state(*, params=True) -> InternalState:
  frame = base.current_frame()
  rng = frame.rng_stack.peek()
  if rng is not None:
    rng = rng.internal_state
  return InternalState(
      params=(copy_structure(frame.params) if params else None),
      state=copy_structure(frame.state),
      rng=copy_structure(rng))


def update_recursive(dst: MutableMapping[Any, Any], src: Mapping[Any, Any]):
  for k, v in src.items():
    if isinstance(v, collections.Mapping):
      dst.setdefault(k, {})
      update_recursive(dst[k], v)
    else:
      if v is not None:
        # NOTE: We only expect `None` values thanks to `difference`.
        dst[k] = v


def update_internal_state(state: InternalState):
  frame = base.current_frame()
  if not frame.params_frozen and state.params is not None:
    update_recursive(frame.params, state.params)
  update_recursive(frame.state, state.state)
  rng = state.rng
  if rng is not None:
    frame.rng_stack.peek().replace_internal_state(rng)


def temporary_internal_state(state: InternalState):
  """Pushes a temporary copy of the internal state."""
  state = copy_structure(state)
  rng = state.rng
  if rng is not None:
    rng = base.PRNGSequence(rng)
  current_state = internal_state()
  params = state.params
  if params is None:
    params = current_state.params
  state = state.state
  if state is None:
    state = current_state.state
  frame = base.current_frame()
  frame = frame.evolve(params=params, state=state, rng=rng)
  return base.frame_stack(frame)


def vjp(
    fun: Callable, *primals, has_aux=False,
) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:
    r"""Creates a function which returns the vjp of ``fun`` as *primals.
    NOTE: You only need this in a very specific case that you want to take a
    vjp **inside** a :func:`transform`\ ed function and the function you are
    differentiating uses :func:`set_state`. For example:
    """

    if not base.inside_transform():
        raise ValueError("hk.vjp() should not be used outside of hk.transform(). "
                         "Use jax.vjp() instead.")

    @functools.wraps(fun)
    def stateful_fun(*args, **kwargs):
        state_in = kwargs.pop("hk_state")
        with temporary_internal_state(state_in):
            out = fun(*args, **kwargs)
            out, aux = (out if has_aux else (out, None))
            state_out = difference(state_in, internal_state())
            return out, (aux, state_out)

    _, vjp_fun = jax.vjp(stateful_fun, *primals, has_aux=True)

    @functools.wraps(vjp_fun)
    def wrapper(*args, **kwargs):
        kwargs["hk_state"] = internal_state()
        (primals, (aux, hk_state)), vjpfun = vjp_fun(*args, **kwargs)
        update_internal_state(hk_state)
        if has_aux:
            return primals, vjpfun, aux
        else:
            return primals, vjpfun

    return wrapper
