import jax
import jax.numpy as jnp
from flax import nnx


class EMA(nnx.Module):
    def __init__(
        self,
        module: nnx.Module,
        update_weight: float,
    ):
        self.module = module

        def _interp(ema_param, update_param):
            return jnp.interp(
                x=update_weight,
                xp=jnp.array([0.0, 1.0]),
                fp=jnp.stack([ema_param, update_param]),
            )

        self._interp = _interp

    def update(self, update_module: nnx.Module):
        old_ema_params = nnx.state(self.module, nnx.Param)
        update_params = nnx.state(update_module, nnx.Param)
        new_ema_params = jax.tree.map(
            self._interp,
            old_ema_params,
            update_params,
        )
        nnx.update(self.module, new_ema_params)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
