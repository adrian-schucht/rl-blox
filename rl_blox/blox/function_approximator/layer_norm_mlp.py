from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.linear import default_bias_init

default_init = jax.nn.initializers.variance_scaling(
    scale=2, mode="fan_avg", distribution="uniform"
)


class LayerNormMLP(nnx.Module):
    """Multilayer Perceptron with layer normalization after each hidden layer.

    Parameters
    ----------
    n_features : int
        Number of features.

    n_outputs : int
        Number of output components.

    hidden_nodes : list
        Numbers of hidden nodes of the MLP.

    activation : str | Callable[[jnp.ndarray], jnp.ndarray]
        Activation function for hidden layers. Has to be the name of a function
        defined in the flax.nnx module or a Callable.

    rngs : nnx.Rngs
        Random number generator.

    kernel_init : nnx.Initializer, optional
        Initializer for the weights of the hidden linear layers.

    output_kernel_init : nnx.Initializer, optional
        Initializer for the weights of the output linear layer.

    bias_init : nnx.Initializer, optional
        Initializer for the biases of all linear layers.
    
    output_activation : str | Callable[[jnp.ndarray], jnp.ndarray] | None, optional
        Activation function for the output layer. If None (default), no
        activation function is used for the output layer. Otherwise, this has to
        be the name of a function defined in the flax.nnx module or a Callable.

    dropout_rates: list[float] | None, optional
        One probability of dropout for each hidden layer (same length as
        hidden_nodes). If None (default), no dropout will happen.
    """

    n_outputs: int
    """Number of output components."""

    activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function for hidden layers."""

    output_activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function for the output layer."""

    hidden_layers: nnx.List
    """Hidden layers."""

    dropout_layers: nnx.List
    """Dropout layers for hidden layers."""

    layer_norms: nnx.List
    """Layer normalization layers for hidden layers."""

    output_layer: nnx.Linear
    """Output layer."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str | Callable[[jnp.ndarray], jnp.ndarray],
        rngs: nnx.Rngs,
        kernel_init: nnx.Initializer = default_init,
        output_kernel_init: nnx.Initializer = default_init,
        bias_init: nnx.Initializer = default_bias_init,
        output_activation: str | Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        dropout_rates: list[float] | None = None,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_nodes)
        else:
            chex.assert_equal_size([hidden_nodes, dropout_rates])

        self.n_outputs = n_outputs
        if isinstance(activation, str):
            self.activation = getattr(nnx, activation)
        else:
            self.activation = activation

        if output_activation is None:
            self.output_activation = nnx.identity
        elif isinstance(output_activation, str):
            self.output_activation = getattr(nnx, activation)
        else:
            self.output_activation = output_activation

        hidden_layers = []
        dropout_layers = []
        layer_norms = []
        n_in = n_features
        for n_out, dropout_rate in zip(
            hidden_nodes, dropout_rates, strict=True
        ):
            hidden_layers.append(
                nnx.Linear(
                    n_in,
                    n_out,
                    rngs=rngs,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                )
            )
            dropout_layers.append(nnx.Dropout(dropout_rate, rngs=rngs))
            layer_norms.append(nnx.LayerNorm(num_features=n_out, rngs=rngs))
            n_in = n_out
        self.layer_norms = nnx.List(layer_norms)
        self.dropout_layers = nnx.List(dropout_layers)
        self.hidden_layers = nnx.List(hidden_layers)

        self.output_layer = nnx.Linear(
            n_in,
            n_outputs,
            rngs=rngs,
            kernel_init=output_kernel_init,
            bias_init=bias_init,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer, dropout, norm in zip(
            self.hidden_layers,
            self.dropout_layers,
            self.layer_norms,
            strict=True,
        ):
            x = self.activation(norm(dropout(layer(x))))
        return self.output_activation(self.output_layer(x))
