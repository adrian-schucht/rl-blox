# Original source code of https://github.com/nicklashansen/tdmpc2 in one file

# Dependencies:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install torchrl tensordict termcolor
# pip install array-api-compat # for gymnasium.wrappers.NumpyToTorch

# MIT License
#
# Copyright (c) Nicklas Hansen (2023).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import os

from recordclass import recordclass

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
import datetime
import random
import time
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, override, Literal

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optax.tree
import pandas as pd
from flax import nnx
from jax import lax, tree_util, Array
from jax.typing import ArrayLike
from termcolor import colored
from tqdm.rich import trange

from rl_blox.logging.logger import LoggerBase
from rl_blox.logging.timer import Timer
from rl_blox.blox.ema import EMA
from rl_blox.blox.losses import mse_loss
from rl_blox.blox.replay_buffer import SubtrajectoryReplayBuffer

AGENT_CHECKPOINTING_ID = "agent"

MODEL_SIZE = {  # parameters (M)
    1: {
        "enc_dim": 256,
        "mlp_dim": 384,
        "latent_dim": 128,
        "num_enc_layers": 2,
        "num_q": 2,
    },
    5: {
        "enc_dim": 256,
        "mlp_dim": 512,
        "latent_dim": 512,
        "num_enc_layers": 2,
        "num_q": 5,
    },
    19: {
        "enc_dim": 1024,
        "mlp_dim": 1024,
        "latent_dim": 768,
        "num_enc_layers": 3,
        "num_q": 5,
    },
    48: {
        "enc_dim": 1792,
        "mlp_dim": 1792,
        "latent_dim": 768,
        "num_enc_layers": 4,
        "num_q": 5,
    },
    317: {
        "enc_dim": 4096,
        "mlp_dim": 4096,
        "latent_dim": 1376,
        "num_enc_layers": 5,
        "num_q": 8,
    },
}

TASK_SET = {
    "mt30": [
        # 19 original dmcontrol tasks
        "walker-stand",
        "walker-walk",
        "walker-run",
        "cheetah-run",
        "reacher-easy",
        "reacher-hard",
        "acrobot-swingup",
        "pendulum-swingup",
        "cartpole-balance",
        "cartpole-balance-sparse",
        "cartpole-swingup",
        "cartpole-swingup-sparse",
        "cup-catch",
        "finger-spin",
        "finger-turn-easy",
        "finger-turn-hard",
        "fish-swim",
        "hopper-stand",
        "hopper-hop",
        # 11 custom dmcontrol tasks
        "walker-walk-backwards",
        "walker-run-backwards",
        "cheetah-run-backwards",
        "cheetah-run-front",
        "cheetah-run-back",
        "cheetah-jump",
        "hopper-hop-backwards",
        "reacher-three-easy",
        "reacher-three-hard",
        "cup-spin",
        "pendulum-spin",
    ],
    "mt80": [
        # 19 original dmcontrol tasks
        "walker-stand",
        "walker-walk",
        "walker-run",
        "cheetah-run",
        "reacher-easy",
        "reacher-hard",
        "acrobot-swingup",
        "pendulum-swingup",
        "cartpole-balance",
        "cartpole-balance-sparse",
        "cartpole-swingup",
        "cartpole-swingup-sparse",
        "cup-catch",
        "finger-spin",
        "finger-turn-easy",
        "finger-turn-hard",
        "fish-swim",
        "hopper-stand",
        "hopper-hop",
        # 11 custom dmcontrol tasks
        "walker-walk-backwards",
        "walker-run-backwards",
        "cheetah-run-backwards",
        "cheetah-run-front",
        "cheetah-run-back",
        "cheetah-jump",
        "hopper-hop-backwards",
        "reacher-three-easy",
        "reacher-three-hard",
        "cup-spin",
        "pendulum-spin",
        # meta-world mt50
        "mw-assembly",
        "mw-basketball",
        "mw-button-press-topdown",
        "mw-button-press-topdown-wall",
        "mw-button-press",
        "mw-button-press-wall",
        "mw-coffee-button",
        "mw-coffee-pull",
        "mw-coffee-push",
        "mw-dial-turn",
        "mw-disassemble",
        "mw-door-open",
        "mw-door-close",
        "mw-drawer-close",
        "mw-drawer-open",
        "mw-faucet-open",
        "mw-faucet-close",
        "mw-hammer",
        "mw-handle-press-side",
        "mw-handle-press",
        "mw-handle-pull-side",
        "mw-handle-pull",
        "mw-lever-pull",
        "mw-peg-insert-side",
        "mw-peg-unplug-side",
        "mw-pick-out-of-hole",
        "mw-pick-place",
        "mw-pick-place-wall",
        "mw-plate-slide",
        "mw-plate-slide-side",
        "mw-plate-slide-back",
        "mw-plate-slide-back-side",
        "mw-push-back",
        "mw-push",
        "mw-push-wall",
        "mw-reach",
        "mw-reach-wall",
        "mw-shelf-place",
        "mw-soccer",
        "mw-stick-push",
        "mw-stick-pull",
        "mw-sweep-into",
        "mw-sweep",
        "mw-window-open",
        "mw-window-close",
        "mw-bin-picking",
        "mw-box-close",
        "mw-door-lock",
        "mw-door-unlock",
        "mw-hand-insert",
    ],
}

CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("episode", "E", "int"),
    ("step", "I", "int"),
    ("episode_reward", "R", "float"),
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
]

CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "eval": "green",
}

AgentConfig = recordclass(
    "AgentConfig",
    [
        "task",
        "obs",
        "batch_size",
        "reward_coef",
        "value_coef",
        "consistency_coef",
        "rho",
        "lr",
        "enc_lr_scale",
        "grad_clip_norm",
        "tau",
        "discount_denom",
        "discount_min",
        "discount_max",
        "mpc",
        "iterations",
        "num_samples",
        "num_elites",
        "num_pi_trajs",
        "horizon",
        "min_std",
        "max_std",
        "temperature",
        "log_std_min",
        "log_std_max",
        "entropy_coef",
        "num_bins",
        "vmin",
        "vmax",
        "model_size",
        "num_enc_layers",
        "enc_dim",
        "num_channels",
        "mlp_dim",
        "latent_dim",
        "num_q",
        "dropout",
        "simnorm_dim",
        "compile",
        "tasks",
        "bin_size",
        "action_dim",
        "episode_length",
        "obs_shape",
    ],
)

TrainingConfig = recordclass(
    "TrainingConfig",
    [
        "eval_episodes",
        "eval_freq",
        "steps",
        "buffer_size",
        "seed_steps",
        "progress_bar",
    ],
)

FullTrainingConfig = recordclass(
    "FullTrainingConfig",
    [  # Fields are specified explicitly to prevent auto code inspection false positives
        "task",
        "obs",
        "batch_size",
        "reward_coef",
        "value_coef",
        "consistency_coef",
        "rho",
        "lr",
        "enc_lr_scale",
        "grad_clip_norm",
        "tau",
        "discount_denom",
        "discount_min",
        "discount_max",
        "mpc",
        "iterations",
        "num_samples",
        "num_elites",
        "num_pi_trajs",
        "horizon",
        "min_std",
        "max_std",
        "temperature",
        "log_std_min",
        "log_std_max",
        "entropy_coef",
        "num_bins",
        "vmin",
        "vmax",
        "model_size",
        "num_enc_layers",
        "enc_dim",
        "num_channels",
        "mlp_dim",
        "latent_dim",
        "num_q",
        "dropout",
        "simnorm_dim",
        "compile",
        "tasks",
        "bin_size",
        "action_dim",
        "episode_length",
        "obs_shape",
        "eval_episodes",
        "eval_freq",
        "steps",
        "buffer_size",
        "seed_steps",
        "progress_bar",
    ],
)


def make_agent_cfg(
    task: str,
    obs: str = "state",
    # Planning
    horizon: int = 3,
    iterations: int = 6,
    num_samples: int = 512,
    num_pi_trajs: int = 24,
    num_elites: int = 64,
    min_std: float = 0.05,
    max_std: float = 2,
    temperature: float = 0.5,
    # momentun: No
    # Policy prior
    log_std_min: float = -10,
    log_std_max: float = 2,
    # Replay buffer (Nothing)
    # capacity: 1_000_000
    # sampling: uniform
    # Architecture
    model_size: int = 5,  # 1, 5, 19, 48, 317
    enc_dim: int = 256,
    num_enc_layers: int = 2,  # additional
    mlp_dim: int = 512,
    latent_dim: int = 512,
    # task embedding dim: 96
    # task embedding norm: 1
    # activation: layernorm + mish
    dropout: float = 0.01,
    num_q: int = 5,
    num_bins: int = 101,
    simnorm_dim: int = 8,
    # simnorm temperature (tau): 1
    # Optimization
    # update to data ratio: 1
    batch_size: int = 256,
    consistency_coef=20,
    reward_coef=0.1,
    value_coef=0.1,
    rho=0.5,  # lambda in the paper
    # q function momentum coef: 0.99
    entropy_coef: float = 1e-4,
    # policy prior loss norm: Moving (5%,95%) percentiles
    # optimizer: Adam
    lr: float = 3e-4,
    enc_lr_scale=0.3,
    grad_clip_norm=20,
    num_channels: int = 32,
    # training
    tau=0.01,
    discount_denom=5,
    discount_min=0.95,
    discount_max=0.995,
    # planning
    mpc: bool = True,
    # actor
    # critic
    vmin: float = -10,
    vmax: float = +10,
    # architecture
    # speedups
    compile: bool = False,
) -> AgentConfig:
    """Create the agent-specific configuration for TD-MPC2.

    Parameters
    ----------
    task : str
        Task name.
    obs : str in ["state", "rgb"]
        Observation type.
    batch_size : int
        Number of trajectories (of length 'horizon') to sample from the
        replay buffer during an agent update.
    reward_coef
        Weight of the reward loss term in the total model loss.
    value_coef
        Weight of the value loss term in the total model loss.
    consistency_coef
        Weight of the consistency loss term in the total model loss.
        The consistency loss term ensures that the learned latent forward
        dynamics are consistent with the encoding of the true successor
        states (which come from the observation of actual dynamics of the
        environment).
    rho
        ~Discount factor in loss calculations; lambda in the paper. It is a
        "constant coefficient that weighs temporally farther time steps less"
        in loss calculations. Should be in (0, 1].
    lr : float
        Learning rate.
    enc_lr_scale : float
        Scaling for the encoder learning rate.
    grad_clip_norm : float
        Clip the gradients during backpropagation.
    tau
        For Polyak averaging. Determines the interpolation factor between
        the current parameters of the target network and the parameters of the
        main network.
    discount_denom
        Denominator in the heuristic for the discount factor, which changes
        in response to the episode length, within the bounds of
        [discount_min,discount_max].
    discount_min
        Minimum value for the discount factor in its heuristic.
    discount_max
        Maximum value for the discount factor in its heuristic.
    mpc
        Whether to use MPC planning (via MPPI) for inference, resorting to
        the learned policy only to determine the terminal value past the
        planning horizon. If False, MPC is bypassed and the learned policy
        determines the next action directly.
    iterations : int
        Number of iterations to optimize plan. We add 2 iterations for large
        action spaces (>= 20 dimensions).
    num_samples : int
        Number of samples for MPC planning.
    num_elites : int
        Number of samples to use for update of search distribution in planning.
    num_pi_trajs : int
        Number of samples generated with policy.
    horizon : int
        Planning horizon.
    min_std : float
        Minimum standard deviation for the Gaussian distribution used in
        trajectory sampling for MPPI.
    max_std : float
        Maximum standard deviation for the Gaussian distribution used in
        trajectory sampling for MPPI.
    temperature : float
        Temperature for planning with MPPI.
    log_std_min : float
        Minimum of log std for actor.
    log_std_max : float
        Maximum of log std for actor.
    entropy_coef : float
        Entropy coefficient for policy update.
    num_bins : int
        Numbers of bins to be used for two-hot encoding (effectively,
        this concerns reward and return.)
    vmin : float
        Natural logarithm of the expected minimum value to be represented by
        two-hot encoding (effectively, this concerns reward and return.)
    vmax : float
        Natural logarithm of the expected maximum value to be represented by
        two-hot encoding (effectively, this concerns reward and return.)
    model_size : int
        Model size, must be either one of [1, 5, 19, 48, 317] or None.
        If none, use values for num_enc_layers, enc_dim, num_channels, mlp_dim,
        latent_dim, and num_q to define model architecture.
    num_enc_layers : int
        Number of layers in encoder.
    enc_dim : int
        Number of nodes in encoder layers.
    num_channels : int
        Number of channels for convolutional encoder with raw image
        observations.
    mlp_dim : int
        Number of hidden nodes in dynamics model, reward model, policy, and Q
        network.
    latent_dim : int
        Dimensions of latent space to which the encoder projects.
    num_q : int
        Number of networks in the ensemble of Q-functions.
    dropout : float
        Dropout probability for Q-functions.
    simnorm_dim : int
        Number of dimensions for simplicial normalization in encoder.
    compile : bool
        Compile graphs for faster training.

    Returns
    -------
    AgentConfiguration
        The agent-specific configuration for TD-MPC2.
    """
    # Model size
    if model_size is not None:
        if model_size not in MODEL_SIZE:
            raise ValueError(
                f"Invalid model size {model_size}. Must be one of {list(MODEL_SIZE.keys())}"
            )
        enc_dim = MODEL_SIZE[model_size]["enc_dim"]
        mlp_dim = MODEL_SIZE[model_size]["mlp_dim"]
        latent_dim = MODEL_SIZE[model_size]["latent_dim"]
        num_enc_layers = MODEL_SIZE[model_size]["num_enc_layers"]
        num_q = MODEL_SIZE[model_size]["num_q"]
    return AgentConfig(
        obs=obs,
        batch_size=batch_size,
        reward_coef=reward_coef,
        value_coef=value_coef,
        consistency_coef=consistency_coef,
        rho=rho,
        lr=lr,
        enc_lr_scale=enc_lr_scale,
        grad_clip_norm=grad_clip_norm,
        tau=tau,
        discount_denom=discount_denom,
        discount_min=discount_min,
        discount_max=discount_max,
        mpc=mpc,
        iterations=iterations,
        num_samples=num_samples,
        num_elites=num_elites,
        num_pi_trajs=num_pi_trajs,
        horizon=horizon,
        min_std=min_std,
        max_std=max_std,
        temperature=temperature,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        entropy_coef=entropy_coef,
        num_bins=num_bins,
        vmin=vmin,
        vmax=vmax,
        model_size=model_size,
        num_enc_layers=num_enc_layers,
        enc_dim=enc_dim,
        num_channels=num_channels,
        mlp_dim=mlp_dim,
        latent_dim=latent_dim,
        num_q=num_q,
        dropout=dropout,
        simnorm_dim=simnorm_dim,
        compile=compile,
        tasks=TASK_SET.get(task, [task]),
        bin_size=None,  # Set during training
        action_dim=None,  # Set during training
        episode_length=None,  # Set during training
        obs_shape=None,  # Set during training
    )


def make_training_cfg(
    # eval
    eval_episodes=10,
    eval_freq=50_000,
    steps=10_000_000,
    # training
    buffer_size=1_000_000,
    progress_bar=True,
) -> TrainingConfig:
    """Create the training-specific configuration for TD-MPC2.

    Parameters
    ----------
    eval_episodes : int
        Number of evaluation episodes.
    eval_freq : int
        Evaluate every eval_freq steps.
    steps : int
        Number of training steps in environment steps
    buffer_size : int
        Size of the replay buffer.
    progress_bar : bool, optional
        Enable the tqdm progressbar? Enabled by default.

    Returns
    -------
    TrainingConfiguration
        The training-specific configuration for TD-MPC2.
    """
    return TrainingConfig(
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        steps=steps,
        buffer_size=buffer_size,
        seed_steps=None,  # None -> heuristic
        progress_bar=progress_bar,
    )


@partial(jax.jit, static_argnames=["vmin", "vmax", "bin_size", "num_bins"])
def soft_ce(
    pred: ArrayLike,
    target: ArrayLike,
    vmin: float,
    vmax: float,
    bin_size: float,
    num_bins: int,
):
    """Compute the cross entropy loss between predictions and soft targets."""
    pred = nnx.log_softmax(pred, axis=-1)
    target = two_hot(target, vmin, vmax, bin_size, num_bins)
    return -jnp.sum(target * pred, axis=-1, keepdims=True)


def safe_log_std(
    x: ArrayLike,
    low: ArrayLike,
    dif: ArrayLike,
):
    return low + 0.5 * dif * (jnp.tanh(x) + 1)


def gaussian_logprob(
    eps: ArrayLike,
    log_std: ArrayLike,
):
    """Compute Gaussian log probability."""
    residual = -0.5 * jnp.pow(eps, 2) - log_std
    log_prob = residual - 0.9189385175704956
    return jnp.sum(log_prob, axis=-1, keepdims=True)


def squash(
    mu: ArrayLike,
    pi: ArrayLike,
    log_pi: ArrayLike,
):
    """Apply squashing function."""
    mu = jnp.tanh(mu)
    pi = jnp.tanh(pi)
    squashed_pi = jnp.log(nnx.relu(1 - jnp.pow(pi, 2)) + 1e-6)
    log_pi = log_pi - jnp.sum(squashed_pi, axis=-1, keepdims=True)
    return mu, pi, log_pi


@jax.jit
def symlog(x: ArrayLike) -> Array:
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


@jax.jit
def symexp(x: ArrayLike) -> Array:
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


# TODO: squeeze x, 1
@partial(jax.jit, static_argnames=["vmin", "vmax", "bin_size", "num_bins"])
def two_hot(
    x: ArrayLike,
    vmin: float,
    vmax: float,
    bin_size: float,
    num_bins: int,
) -> Array:
    """Convert array of scalars to soft two-hot encoded targets for discrete
    regression"""
    if num_bins == 0:
        return jnp.array(x)
    elif num_bins == 1:
        return symlog(x)
    x = jnp.clip(symlog(x), vmin, vmax)
    bin_idx = jnp.floor((x - vmin) / bin_size)
    bin_offset = (x - vmin) / bin_size - bin_idx
    soft_two_hot = jnp.zeros((x.shape[0], num_bins), dtype=x.dtype)
    soft_two_hot = soft_two_hot.at[
        jnp.arange(x.shape[0]), bin_idx.astype(dtype=int)
    ].set(1 - bin_offset)
    soft_two_hot = soft_two_hot.at[
        jnp.arange(x.shape[0]), bin_idx.astype(dtype=int) + 1
    ].set(bin_offset)
    return soft_two_hot


@partial(jax.jit, static_argnames=["vmin", "vmax", "num_bins"])
def two_hot_inv(
    x: ArrayLike,
    vmin: float,
    vmax: float,
    num_bins: int,
):
    """Convert an array of soft two-hot encoded vectors to an array of
    scalars."""
    if num_bins == 0:
        return jnp.array(x)
    elif num_bins == 1:
        return symexp(x)
    x = nnx.softmax(x, axis=-1)
    dreg_bins = jnp.linspace(vmin, vmax, num_bins, dtype=x.dtype)
    x = jnp.sum(x * dreg_bins, axis=-1)
    return symexp(x)


def gumbel_softmax_sample(
    p: ArrayLike,
    rngs: nnx.Rngs,  # TODO: PRNGKey
    temperature: float = 1.0,
    dim: int = 0,
):
    logits = jnp.log(p)
    # Generate Gumbel noise
    gumbels = -jnp.log(  # TODO: note torch.legacy_contiguous_format
        rngs.exponential(
            shape=logits.shape,
            dtype=logits.dtype,
        )
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = nnx.softmax(gumbels, axis=dim)
    return jnp.argmax(y_soft, axis=-1)


class NonlearnableVariable(nnx.Variable):
    pass


class RunningScale(nnx.Module):
    """Running trimmed scale estimator."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.value = NonlearnableVariable(jnp.array(1.0))
        self._percentiles = NonlearnableVariable(jnp.array([5, 95]))

    # TODO: removable?
    # def state_dict(self):
    #     return dict(value=self.value, percentiles=self._percentiles)
    #
    # def load_state_dict(self, state_dict):
    #     self.value.copy_(state_dict["value"])
    #     self._percentiles.copy_(state_dict["percentiles"])

    def _positions(self, x_shape) -> tuple[Array, Array, Array, Array]:
        positions = self._percentiles * (x_shape - 1) / 100
        floored = jnp.floor(positions)
        ceiled = floored + 1.0
        ceiled = jnp.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        return (
            floored.astype(jnp.int64),
            ceiled.astype(jnp.int64),
            jnp.expand_dims(weight_floored, axis=1),
            jnp.expand_dims(weight_ceiled, axis=1),
        )

    def _percentile(self, x: Array) -> Array:
        x_dtype, x_shape = x.dtype, x.shape
        x = jax.vmap(jnp.ravel)(x)
        # x = x.flatten(1, x.ndim - 1) # NOTE: this was here before
        in_sorted = jnp.sort(x, axis=0)
        floored, ceiled, weight_floored, weight_ceiled = self._positions(
            x.shape[0]
        )
        d0 = in_sorted[floored] * weight_floored
        d1 = in_sorted[ceiled] * weight_ceiled
        return jnp.reshape(d0 + d1, (-1,) + x_shape[1:]).astype(x_dtype)
        # NOTE: was previously:
        # return (d0 + d1).reshape(-1, *x_shape[1:]).to(x_dtype)

    def update(self, x: Array):
        percentiles = self._percentile(x)  # NOTE: previously detach()
        value = jnp.clip(percentiles[1] - percentiles[0], min=1.0)
        self.value = jnp.interp(
            x=self.cfg.tau,
            xp=jnp.array([0.0, 1.0]),
            fp=jnp.array([self.value, value]),
        )

    def __call__(self, x: Array, update=False):
        if update:
            self.update(x)
        return x / self.value

    # TODO: removable?
    # def __repr__(self):
    #     return f"RunningScale(S: {self.value})"


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


class DefaultSuccessInfoWrapper(gym.Wrapper):
    """
    Gym environment wrapper for filling the info["success"] field with a default value if missing.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, termination, truncation, info = self.env.step(action)
        info = defaultdict(float, info)
        info["success"] = float(info["success"])
        return obs, reward, termination, truncation, info


# TODO: removable?
class Buffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg: FullTrainingConfig):
        self.cfg = cfg
        self._device = torch.device("cuda:0")
        self._capacity = min(cfg.buffer_size, cfg.steps) # transitions
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
            cache_values=False,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon + 1) # batch: (partial?) trajectories
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f"Buffer capacity: {self._capacity:,}")
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in tds.values()
            ]
        ) / len(tds)
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes / 1e9:.2f} GB")
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = "cuda:0" if 2.5 * total_bytes < mem_free else "cpu"
        print(f"Using {storage_device.upper()} memory for storage.")
        self._storage_device = torch.device(storage_device)
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device)
        )

    def load(self, td):
        """
        Load a batch of episodes into the buffer. This is useful for loading data from disk,
        and is more efficient than adding episodes one by one.
        """
        num_new_eps = len(td)
        episode_idx = torch.arange(
            self._num_eps, self._num_eps + num_new_eps, dtype=torch.int64
        )
        td["episode"] = episode_idx.unsqueeze(-1).expand(
            -1, td["reward"].shape[1]
        )
        if self._num_eps == 0:
            self._buffer = self._init(td[0])
        td = td.reshape(td.shape[0] * td.shape[1])
        self._buffer.extend(td)
        self._num_eps += num_new_eps
        return self._num_eps

    def add(self, td):
        """Add an episode to the buffer."""
        td["episode"] = torch.full_like(
            td["reward"], self._num_eps, dtype=torch.int64
        )
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        td = td.select("obs", "action", "reward", "task", strict=False).to(
            self._device, non_blocking=True
        )
        obs = td.get("obs").contiguous()
        action = td.get("action")[1:].contiguous()
        reward = td.get("reward")[1:].unsqueeze(-1).contiguous()
        task = td.get("task", None)
        if task is not None:
            task = task[0].contiguous()
        return obs, action, reward, task

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(td)


class OnlineTrainer:
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(
        self,
        cfg: FullTrainingConfig,
        env: gym.Env[gym.spaces.Box, gym.spaces.Box],
        agent: TDMPC2,
        logger: LoggerBase | None = None,
        timer: Timer = Timer(),
    ):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = SubtrajectoryReplayBuffer(
            buffer_size=min(cfg.buffer_size, cfg.steps),
            horizon=cfg.horizon,
        )
        self.logger = logger
        self.timer = timer
        #print("Architecture:", self.agent.model)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time.time()
        self._transitions_in_episode = []

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time.time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, _ = self.env.reset()
            done, ep_reward, t = False, 0, 0
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, termination, truncation, info = self.env.step(
                    action
                )
                done = termination or truncation
                ep_reward += reward
                t += 1
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )


    # TODO: removable?
    # def to_td(self, obs, action=None, reward=None):
    #     """Creates a TensorDict for a new episode."""
    #     if isinstance(obs, dict):
    #         obs = TensorDict(obs, batch_size=(), device="cpu")
    #     else:
    #         obs = obs.unsqueeze(0).cpu()
    #     if action is None:
    #         action = torch.full_like(
    #             self.env.sample_action_space(), float("nan")
    #         )
    #     if reward is None:
    #         reward = torch.tensor(float("nan"))
    #     td = TensorDict(
    #         obs=obs,
    #         action=action.unsqueeze(0),
    #         reward=reward.unsqueeze(0),
    #         batch_size=(1,),
    #     )
    #     return td

    def train(self):
        """Train a TD-MPC2 agent."""
        done, eval_next = True, False
        steps_in_episode = 0
        episode_reward = 0.0
        progress = trange(
            self._step, self.cfg.steps, disable=not self.cfg.progress_bar
        )
        self.timer.start("training")
        self.timer.start("seed_acquisition")
        for self._step in np.arange(self._step, self.cfg.steps + 1):
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = False  # FIXME: originally True

            # Reset environment
            if done:
                if eval_next:
                    self.timer.start("eval")
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    # TODO: log? evaluate at all?
                    eval_next = False
                    self.timer.stop("eval")

                if self._step > 0:
                    episode_success = info["success"]
                    if self.logger is not None:
                        self.logger.record_stat("return", value=episode_reward)
                        self.logger.record_stat(
                            "success", value=episode_success
                        )
                        self.logger.stop_episode(steps_in_episode)

                    steps_in_episode = 0
                    episode_reward = 0.0
                    self._ep_idx += 1

                if self.logger is not None:
                    self.logger.start_new_episode()

                obs, _ = self.env.reset()

            # Collect experience
            if self._step > self.cfg.seed_steps:
                self.timer.start("agent_act")
                t0 = (steps_in_episode == 0)
                action = self.agent.act(obs, t0=t0)
                self.timer.stop("agent_act")
            else:
                self.timer.start("env_sample_action_space")
                action = self.env.action_space.sample() # TODO
                self.timer.stop("env_sample_action_space")
            prev_obs = obs
            self.timer.start("env_step")
            obs, reward, termination, truncation, info = self.env.step(action)
            self.timer.stop("env_step")
            done = termination or truncation
            _ = self.buffer.add_sample(
                observation=prev_obs,
                action=action,
                reward=reward,
                next_observation=obs, # TODO: removable? Was not here in original impl
                terminated=termination,
                truncated=truncation,
            )
            steps_in_episode += 1
            episode_reward += reward

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    self.timer.stop("seed_acquisition")
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                for i in range(num_updates):
                    self.timer.start("agent_update")
                    metrics = self.agent.update(self.buffer)
                    self.timer.stop("agent_update")
                    progress.update()  # 1 update = 1 step, just not necessarily synchronously
                    if i == num_updates - 1:
                        for k, v in metrics.items():
                            self.logger.record_stat(k, v)
                self.logger.record_epoch(
                    AGENT_CHECKPOINTING_ID, self.agent, step=self._step
                )

        # End last (potentially partial) episode # TODO: necessary?
        if self.logger is not None:
            self.timer.stop("training")
            self.timer.log(self.logger)
            self.logger.stop_episode(steps_in_episode)


class TDMPC2(nnx.Module):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    @staticmethod
    def from_config(cfg: AgentConfig, rng_seed: int):
        rngs = nnx.Rngs(rng_seed)
        np_rng = np.random.default_rng(rng_seed)
        model = WorldModel(cfg, rngs)
        # TODO: Ensure model._pi is actually masked out by this.
        model_optim = nnx.Optimizer(
            model,
            optax.adam(
                learning_rate=cfg.lr * cfg.enc_lr_scale
            ),
            wrt=nnx.Param,
        )
        print("ATTENTION: Correct Optimizer!!!")
        # model_optim = nnx.Optimizer(
        #     model,
        #     optax.chain(
        #         optax.clip_by_global_norm(cfg.grad_clip_norm), # TODO: This ok?
        #         optax.partition(
        #             {
        #                 "_encoder": optax.adam(
        #                     learning_rate=cfg.lr * cfg.enc_lr_scale
        #                 ),
        #                 "_dynamics": optax.adam(learning_rate=cfg.lr),
        #                 "_reward": optax.adam(learning_rate=cfg.lr),
        #                 "_Qs": optax.adam(learning_rate=cfg.lr),
        #             },
        #             param_labels=("_encoder", "_dynamics", "_reward", "_Qs"),
        #         ),
        #     ),
        #     wrt=nnx.Param,
        # )
        # TODO: removable? / why []?
        # optim = torch.optim.Adam(
        #     [
        #         {
        #             "params": model._encoder.parameters(),
        #             "lr": cfg.lr * cfg.enc_lr_scale,
        #         },
        #         {"params": model._dynamics.parameters()},
        #         {"params": model._reward.parameters()},
        #         {"params": model._Qs.parameters()},
        #         {"params": []},
        #     ],
        #     lr=cfg.lr,
        #     capturable=True,
        # )
        pi_optim = nnx.Optimizer(
            model._pi,
            optax.chain(
                optax.clip_by_global_norm(cfg.grad_clip_norm), # TODO: This ok?
                optax.adam(
                    learning_rate=cfg.lr,
                    eps=1e-5,
                ),
            ),
            wrt=nnx.Param,
        )
        model.eval()
        scale = RunningScale(cfg)
        discount = TDMPC2._get_discount(cfg.episode_length, cfg)
        _prev_mean = NonlearnableVariable(
            jnp.zeros(shape=(cfg.horizon, cfg.action_dim))
        )
        return TDMPC2(
            model,
            model_optim,
            pi_optim,
            scale,
            _prev_mean,
            cfg,
            discount,
            rngs,
            np_rng,
        )

    def __init__(
        self,
        model: WorldModel,
        model_optim: nnx.Optimizer,
        pi_optim: nnx.Optimizer,
        scale: RunningScale,
        _prev_mean: NonlearnableVariable,
        cfg: AgentConfig,
        discount: Array,
        rngs: nnx.Rngs,
        np_rng: np.random.Generator,
    ):
        self.cfg = cfg
        self.model = model
        self.model_optim = model_optim
        self.pi_optim = pi_optim
        self.scale = scale
        self.discount = discount
        self._prev_mean = _prev_mean
        self._rngs = rngs
        self._np_rng = np_rng

    def _tree_flatten(self):
        # dynamic
        children = (
            self.model,
            self.model_optim,
            self.pi_optim,
            self.scale,
            self._prev_mean,
        )
        # static
        aux_data = {
            "cfg": self.cfg,
            "discount": self.discount,
        }
        return children, aux_data

    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @staticmethod
    def _get_discount(episode_length: int, cfg: AgentConfig) -> float:
        """Returns the discount factor for a given episode length.

        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed
        as needed.

        Args
        ----
        episode_length
            Length of the episode. Assumes episodes are of fixed length.

        cfg
            AgentConfig used for its discount information.

        Returns
        -------
        float
            Discount factor for the task.
        """
        frac = episode_length / cfg.discount_denom
        return min(
            max(
                (frac - 1) / (frac),
                cfg.discount_min,
            ),
            cfg.discount_max,
        )

    # TODO: removable?
    # def save(self, fp: str):
    #     """
    #     Save state dict of the agent to filepath.
    #
    #     Args
    #     ----
    #     fp
    #         Filepath to save state dict to.
    #     """
    #     torch.save({"model": self.model.state_dict()}, fp)
    #
    # def load(self, fp):
    #     """
    #     Load a saved state dict from filepath (or dictionary) into current agent.
    #
    #     Args:
    #             fp (str or dict): Filepath or state dict to load.
    #     """
    #     if isinstance(fp, dict):
    #         state_dict = fp
    #     else:
    #         state_dict = torch.load(
    #             fp, map_location=torch.get_default_device(), weights_only=False
    #         )
    #     state_dict = (
    #         state_dict["model"] if "model" in state_dict else state_dict
    #     )
    #     state_dict = api_model_conversion(self.model.state_dict(), state_dict)
    #     self.model.load_state_dict(state_dict)
    #     return

    # TODO: lax.stop_gradient() on call
    # @torch.no_grad()
    @partial(jax.jit, static_argnames=["eval_mode", "task"])
    def act(
        self,
        obs: ArrayLike,
        rngs: nnx.Rngs,
        t0: bool = False,
        eval_mode: bool = False,
        task: int | Array | None = None,
    ) -> Array:
        """Select an action by planning in the latent space of the world model.

        Parameters
        ----------
        obs
            Observation from the environment.
        t0
            Whether this is the first observation in the episode.
        eval_mode
            Whether to use the mean of the action distribution.
        task
            Task index (only used for multi-task experiments).

        Returns
        -------
        torch.Tensor
            Action to take in the environment.
        """
        obs = jnp.expand_dims(obs, axis=0)
        if task is not None:
            task = jnp.array([task])
        if self.cfg.mpc:
            return self._plan(obs, t0=t0, eval_mode=eval_mode, task=task)
        z = self.model.encode(obs, task)
        action, info = self.model.pi(z, task, rngs)
        if eval_mode:
            action = info["mean"]
        return action.at[0].get()

    # TODO: removable?
    # @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and
        executing given actions.

        (ll. 5-9, Algorithm 1, TD-MPC (inference), [2]_)
        """
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = two_hot_inv(
                self.model.reward(z, actions[t], task),
                self.cfg.vmin,
                self.cfg.vmax,
                self.cfg.num_bins,
            )
            z = self.model.next(z, actions[t], task)
            G = G + discount * reward
            discount_update = self.discount
            discount = discount * discount_update
        action, _ = self.model.pi(z, task)
        return G + discount * self.model.Q(z, action, rngs=rngs, return_type="avg")

    # TODO: lax.stop_gradient() on call
    # @torch.no_grad()
    @partial(jax.jit, static_argnames=["eval_mode", "task"])
    def _plan(
        self,
        obs: Array,
        rngs: nnx.Rngs,
        t0: bool = False,
        eval_mode: bool = False,
        task: Array | None = None,
    ):
        """Plan a sequence of actions using the learned world model.

        Parameters
        ----------
        obs
            State whose latent representation to plan from.
        t0
            Whether this is the first observation in the episode.
        eval_mode
            Whether to use the mean of the action distribution.
        task
            Task index (only used for multi-task experiments).

        Returns
        -------
        Array
            Action to take in the environment.
        """
        # Sample policy trajectories.
        # (l. 4, Algorithm 1, TD-MPC (inference), [2]_)
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = jnp.empty(
                shape=(
                    self.cfg.horizon,
                    self.cfg.num_pi_trajs,
                    self.cfg.action_dim,
                )
            )
            _z = jnp.repeat(
                jnp.expand_dims(z, 0),
                repeats=self.cfg.num_pi_trajs,
                axis=0,
            )
            # NOTE: before this:
            #_z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                action, _ = self.model.pi(_z, task)
                pi_actions = pi_actions.at[t].set(action)
                _z = self.model.next(_z, pi_actions[t], task)
            action, _ = self.model.pi(_z, task)
            pi_actions[-1] = action

        # Initialize state and parameters
        z = jnp.repeat(
            jnp.expand_dims(z, 0),
            repeats=self.cfg.num_samples,
            axis=0,
        )
        # NOTE: before this:
        # z = z.repeat(self.cfg.num_samples, 1)
        mean = jnp.zeros(
            (self.cfg.horizon, self.cfg.action_dim)
        )
        std = jnp.full(
            (self.cfg.horizon, self.cfg.action_dim),
            fill_value=self.cfg.max_std,
            dtype=jnp.float_,
        )
        mean = mean.at[:-1].set(
            lax.cond(
                t0,
                lambda a: a[0],
                lambda a: a[1],
                (mean.at[:-1].get(), self._prev_mean.at[1:].get()),
            )
        )
        actions = jnp.empty(
            shape=(
                self.cfg.horizon,
                self.cfg.num_samples,
                self.cfg.action_dim,
            ),
        )
        if self.cfg.num_pi_trajs > 0:
            actions = actions.at[:, : self.cfg.num_pi_trajs].set(pi_actions)

        # Iterate MPPI
        # (ll. 2-10, Algorithm 1, TD-MPC (inference), [2]_)
        for _ in range(self.cfg.iterations):
            # Sample MPPI actions
            # (l. 3, Algorithm 1, TD-MPC (inference), [2]_)
            r = rngs.normal(
                shape=(
                    self.cfg.horizon,
                    self.cfg.num_samples - self.cfg.num_pi_trajs,
                    self.cfg.action_dim,
                )
            )
            actions_sample = jnp.expand_dims(mean, 1) + jnp.expand_dims(std, 1) * r
            actions_sample = jnp.clip(actions_sample, -1.0, 1.0)
            action = actions.at[:, self.cfg.num_pi_trajs:].set(actions_sample)

            # Compute elite actions
            # TODO: removable?: .nan_to_num(0)
            value = self._estimate_value(z, actions, task)
            elite_idxs = jnp.argpartition(
                value.squeeze(1),
                self.cfg.num_elites,
                axis=0,
            )
            elite_value = value.at[elite_idxs].get()
            elite_actions = actions.at[:, elite_idxs].get()

            # Update parameters
            # (l. 10, Algorithm 1, TD-MPC (inference), [2]_)
            max_value = jnp.max(elite_value, axis=0)
            score = jnp.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / jnp.sum(score, axis=0)
            mean = (
                jnp.sum(
                    jnp.expand_dims(score, 0) * elite_actions,
                    axis=1,
                ) / (
                    jnp.sum(score, 0) + 1e-9
                )
            )
            std = jnp.sqrt(
                jnp.sum(
                    (
                        jnp.expand_dims(score, 0)
                        * (elite_actions - jnp.expand_dims(mean, 1)) ** 2
                    ),
                    axis=1,
                ) / (jnp.sum(score, 0) + 1e-9)
            )
            std = jnp.clip(std, self.cfg.min_std, self.cfg.max_std)

        # Select action
        # (l. 11, Algorithm 1, TD-MPC (inference), [2]_)
        rand_idx = gumbel_softmax_sample(
            score.squeeze(1),
            rngs,
        )
        actions = jnp.squeeze(jnp.take(elite_actions, rand_idx, axis=1), axis=1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * rngs.normal(self.cfg.action_dim)
        self._prev_mean = mean
        return jnp.clip(a, -1.0, 1.0)

    def _pi_loss(
        self,
        zs: Array,
        task: Array,
        rngs: nnx.Rngs,
    ) -> tuple[Array, dict[str, Array]]:
        """Calculate the loss of the policy on a sequence of latent states.

        (Equation 4, Policy objective, [2]_)

        Parameters
        ----------
        zs
            Sequence of latent states.
        task
            Task index (only used for multi-task experiments).

        Returns
        -------
        tuple[Array, dict[str, Array]]
            A pair of (1) the policy loss and (2) the info dict from
            WorldModel.pi().
        """
        action, info = self.model.pi(zs, task, rngs)
        qs = self.model.Q(zs, action, rngs=rngs, return_type="avg", detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        # (rho is lambda in Equation (4), [2]_)
        rho = jnp.pow(self.cfg.rho, jnp.arange(len(qs)))
        pi_loss = (
            -(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(
                axis=(1, 2)
            )
            * rho
        ).mean()
        return pi_loss, info

    def update_pi(self, zs: Array, task: Array, rngs: nnx.Rngs):
        """Update the policy using a sequence of latent states.

        Parameters
        ----------
        zs
            Sequence of latent states.
        task
            Task index (only used for multi-task experiments).

        Returns
        -------
        float
            Loss of the policy update.

        See Also
        --------
        _pi_loss
        """
        (pi_loss, info), pi_loss_grads = nnx.value_and_grad(
            self._pi_loss,
            has_aux=True,
        )(
            zs, task, rngs
        )
        # TODO: use optax.tree.norm() on the updates from the orbax optimizer.
        # For this, the updates have to be made accessible.
        # Currently, they are hidden behind the facade of nnx.Optimizer.
        #pi_loss.backward()
        #pi_grad_norm = torch.nn.utils.clip_grad_norm_(
        #    self.model._pi.parameters(), self.cfg.grad_clip_norm
        #)
        pi_grad_norm = jnp.zeros(0)
        self.pi_optim.update(self.model.pi, pi_loss_grads)
        # self.pi_optim.zero_grad(set_to_none=True) # TODO: removable?

        info = {
            "policy loss": pi_loss,
            "policy grad norm": pi_grad_norm,
            "policy entropy": info["entropy"],
            "policy scaled entropy": info["scaled_entropy"],
            "policy scale": self.scale.value,
        }
        return info

    # TODO: removable?
    # @torch.no_grad()
    def _td_target(
        self,
        next_z: Array,
        reward: Array,
        task: Array,
        rngs: nnx.Rngs,
    ) -> Array:
        """Compute the TD-target from a reward and the observation at
        the following time step.

        Arguments
        ---------
        next_z
            Latent state at the following time step.
        reward
            Reward at the current time step.
        task
            Task index (only used for multi-task experiments).
        rngs
            Rngs for invoking the policy.

        Returns
        -------
        Array
            TD-target.
        """
        action, _ = self.model.pi(next_z, task, rngs)
        return reward + self.discount * self.model.Q(
            next_z, action, rngs=rngs, return_type="min", target=True
        )

    def _model_loss(
        self,
        obs: Array,
        action: Array,
        reward: Array,
        rngs: nnx.Rngs,
        task=None,
    ) -> tuple[Array, dict[str, Array]]:
        # Compute targets
        next_z = lax.stop_gradient(self.model.encode(obs[1:], task))
        td_targets = lax.stop_gradient(self._td_target(next_z, reward, task, rngs))

        # Prepare for update
        self.model.train()

        # Latent rollout
        zs = jnp.zeros((
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
        ))
        z = self.model.encode(obs.at[0].get(), task)
        zs = zs.at[0].set(z)
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(
            jnp.unstack(action, axis=0),
            jnp.unstack(next_z, axis=0),
            strict=False,
        )):
            z = self.model.next(z, _action, task)
            consistency_loss = (
                consistency_loss + mse_loss(z, _next_z) * self.cfg.rho**t
            )
            zs = zs.at[t + 1].set(z)

        # Predictions
        _zs = zs.at[:-1].get()
        qs = self.model.Q(_zs, action, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)

        # Compute losses
        reward_loss = jnp.float_(0)
        value_loss = jnp.float_(0)
        for t, (
            rew_pred_unbind,
            rew_unbind,
            td_targets_unbind,
            qs_unbind,
        ) in enumerate(
            zip(
                jnp.unstack(reward_preds, axis=0),
                jnp.unstack(reward, axis=0),
                jnp.unstack(td_targets, axis=0),
                jnp.unstack(qs, axis=1),
                strict=False,
            )
        ):
            reward_loss = (
                reward_loss
                + soft_ce(
                    rew_pred_unbind,
                    rew_unbind,
                    self.cfg.vmin,
                    self.cfg.vmax,
                    self.cfg.bin_size,
                    self.cfg.num_bins,
                ).mean()
                * self.cfg.rho**t
            )
            for _, qs_unbind_unbind in enumerate(jnp.unstack(qs_unbind, axis=0)):
                value_loss = (
                    value_loss
                    + soft_ce(
                        qs_unbind_unbind,
                        td_targets_unbind,
                        self.cfg.vmin,
                        self.cfg.vmax,
                        self.cfg.bin_size,
                        self.cfg.num_bins,
                    ).mean()
                    * self.cfg.rho**t
                )

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )

        info = {
            "consistency loss": consistency_loss,
            "reward loss": reward_loss,
            "q loss": value_loss,
            "total loss": total_loss,
        }

        return total_loss, (info, zs)


    def _update(
        self,
        obs: Array,
        action: Array,
        reward: Array,
        # TODO: removable?
        rngs: nnx.Rngs,
        task=None,
    ):

        # Update model
        (model_loss, (model_info, zs)), model_loss_grads = nnx.value_and_grad(
            self._model_loss,
            has_aux=True,
        )(
            obs, action, reward, rngs, task
        )
        # TODO: use optax.tree.norm() on the updates from the orbax optimizer.
        # For this, the updates have to be made accessible.
        # Currently, they are hidden behind the facade of nnx.Optimizer.
        # model_loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(
        #     self.model.parameters(), self.cfg.grad_clip_norm
        # )
        model_grad_norm = jnp.zeros(0)
        self.model_optim.update(self.model, model_loss_grads)
        # self.optim.zero_grad(set_to_none=True) # TODO: removable?

        # Update policy
        # TODO: do something about potentially task==None
        pi_info = self.update_pi(zs, task, rngs)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        info = model_info
        info["grad norm"] = model_grad_norm
        info.update(pi_info)
        mean_info = jax.tree.map(lambda val: jnp.mean(val), info)
        return mean_info

    @staticmethod
    def _prepare_batch(batch):
        # shapes are ~(trajectories, transitions, ...)
        obs, action, reward, next_obs, terminated, truncated = batch
        # make them ~(transitions, trajectories, ...)
        # TODO: improve?
        obs = jnp.swapaxes(obs, 0, 1)
        action = jnp.swapaxes(action, 0, 1)
        reward = jnp.swapaxes(reward, 0, 1)
        next_obs = jnp.swapaxes(next_obs, 0, 1)
        # terminated + truncated are not needed by TDMPC2._update()

        # TDMPC2._update() just needs a single obs sequence, not both
        # obs and next_obs, which share all the same observations but
        # one at the start and one at the end of a trajectory. Thus,
        # combine them to obtain an obs sequence that is 1 longer than
        # the trajectory.
        # shape of obs will then be ~(transitions+1, trajectories, ...)
        obs = jnp.concatenate([jnp.expand_dims(obs.at[0].get(), 0), next_obs])
        return obs, action, reward

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
                buffer (common.buffer.Buffer): Replay buffer.

        Returns:
                dict: Dictionary of training statistics.
        """
        batch = buffer.sample_batch(
            batch_size=self.cfg.batch_size,
            horizon=self.cfg.horizon,
            include_intermediate=True,
            rng=self._np_rng,
        )
        prepared_batch = TDMPC2._prepare_batch(batch)
        obs, action, reward = prepared_batch
        task = None # TODO: clean up
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        return self._update(obs, action, reward, rngs=self._rngs, **kwargs)

# TODO: removable?
# tree_util.register_pytree_node(
#     TDMPC2,
#     TDMPC2._pytree__flatten,
#     TDMPC2._pytree__unflatten,
# )


class WorldModel(nnx.Module):
    """TD-MPC2 implicit world model architecture.

    The world model consists of

    * an encoder that maps observations to latent states
    * a dynamics model that predicts the next latent state given the current
      latent state and action
    * a reward model that predicts the reward given the current latent state
    * a policy prior that predicts the action given the current latent state
    * a Q-function ensemble that predicts the value of a given action
    """

    def __init__(self, cfg: AgentConfig, rngs: nnx.Rngs):
        super().__init__()
        self.cfg = cfg
        self._encoder = enc(cfg, rngs)
        self._dynamics = mlp(
            in_dim=cfg.latent_dim + cfg.action_dim,
            mlp_dims=2 * [cfg.mlp_dim],
            out_dim=cfg.latent_dim,
            rngs=rngs,
            act=SimNorm(cfg.simnorm_dim),
        )
        self._reward = mlp(
            in_dim=cfg.latent_dim + cfg.action_dim,
            mlp_dims=2 * [cfg.mlp_dim],
            out_dim=max(cfg.num_bins, 1),
            rngs=rngs,
        )
        self._reward.layers[-1].kernel = jnp.full_like(
            self._reward.layers[-1].kernel,
            0.0,
        )
        self._pi = mlp(
            in_dim=cfg.latent_dim,
            mlp_dims=2 * [cfg.mlp_dim],
            out_dim=2 * cfg.action_dim,
            rngs=rngs,
        )
        # TODO: Fix Ensemble and following lines (via vmap?)
        self._Qs = Ensemble(
            [
                mlp(
                    in_dim=cfg.latent_dim + cfg.action_dim,
                    mlp_dims=2 * [cfg.mlp_dim],
                    out_dim=max(cfg.num_bins, 1),
                    rngs=rngs,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        # Comes from zero_([..., self._Qs.params["2", "weight"]]):
        for module in self._Qs.modules:
            module.layers[-1].kernel = jnp.full_like(
                module.layers[-1].kernel,
                0.0
            )
        self._detach_Qs = nnx.clone(self._Qs)
        self._target_Qs = EMA(nnx.clone(self._Qs), update_weight=self.cfg.tau)

        self._log_std_min = NonlearnableVariable(jnp.array(cfg.log_std_min))
        self._log_std_dif = NonlearnableVariable(
            jnp.array(cfg.log_std_max) - self._log_std_min
        )
        self.init()

    def init(self):
        # TODO: removable?
        # # Create params
        # self._detach_Qs_params = TensorDictParams(
        #     self._Qs.params.data, no_convert=True
        # )
        # self._target_Qs_params = TensorDictParams(
        #     self._Qs.params.data.clone(), no_convert=True
        # )
        #
        # # Create modules
        # with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
        #     self._detach_Qs = deepcopy(self._Qs)
        #     self._target_Qs = deepcopy(self._Qs)
        #
        # # Assign params to modules
        # # We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
        # delattr(self._detach_Qs, "params")
        # self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        # delattr(self._target_Qs, "params")
        # self._target_Qs.__dict__["params"] = self._target_Qs_params
        pass

    # TODO: removable?
    # def __repr__(self):
    #     repr = "TD-MPC2 World Model\n"
    #     modules = [
    #         "Encoder",
    #         "Dynamics",
    #         "Reward",
    #         "Policy prior",
    #         "Q-functions",
    #     ]
    #     for i, m in enumerate(
    #         [self._encoder, self._dynamics, self._reward, self._pi, self._Qs]
    #     ):
    #         repr += f"{modules[i]}: {m}\n"
    #     repr += f"Learnable parameters: {self.total_params:,}"
    #     return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # TODO: removable?
    # def to(self, *args, **kwargs):
    #     super().to(*args, **kwargs)
    #     self.init()
    #     return self

    # TODO
    @override
    def train(self, **attributes):
        """Overriding `train` method to keep target Q-networks in eval mode."""
        # TODO: Ensure that this behaves the same as torch's train(mode) here.
        super().train(**attributes)
        self._target_Qs.eval()

    def soft_update_target_Q(self):
        """Soft-update target Q-networks using Polyak averaging."""
        self._target_Qs.update(self._detach_Qs)

    # TODO: unused?
    def task_emb(self, x, task):
        """Continuous task embedding for multi-task experiments.

        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = jnp.array([task])
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs: ArrayLike, task):
        """Encodes an observation into its latent representation.

        This implementation assumes a single state-based observation.
        """
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return jnp.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z: ArrayLike, a: ArrayLike, task) -> Array:
        """Predicts the next latent state given the current latent state
        and action.

        Latent dynamics. In the paper: d(z,a,e).
        """
        z = jnp.concat([z, a], axis=-1)
        return self._dynamics(z)

    def reward(self, z: ArrayLike, a: ArrayLike, task) -> Array:
        """Predicts instantaneous (single-step) reward.

        Reward. In the paper: R(z,a,e).
        """
        z = jnp.concat([z, a], axis=-1)
        return self._reward(z)

    def pi(
        self, z: ArrayLike, task, rngs: nnx.Rngs
    ) -> tuple[Array, dict[str, Array]]:
        """Samples an action from the policy prior.

        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        # Gaussian policy prior
        # NOTE: chunk->split, last chunk would be allowed shorter
        mean, log_std = jnp.split(self._pi(z), 2, axis=-1)
        log_std = safe_log_std(log_std, self._log_std_min, self._log_std_dif)
        eps = rngs.normal(shape=mean.shape, dtype=mean.dtype)

        log_prob = gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1]
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * jnp.exp(log_std)
        mean, action, log_prob = squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "log_std": log_std,
            "action_prob": jnp.array(1.0),
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def Q(
        self,
        z: Array,
        a: Array,
        return_type: Literal["min", "avg", "all"]="min",
        rngs: nnx.Rngs | None = None,
        target: bool=False,
        detach: bool=False
    ):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        z = jnp.concatenate([z, a], axis=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        if rngs is None:
            raise RuntimeError(
                f"Cannot calculate Q: Return type is {return_type},"
                + " but no nnx.Rngs were given."
            )

        q_idx = rngs.permutation(self.cfg.num_q)[:2]
        q_value = two_hot_inv(
            out.at[q_idx].get(), self.cfg.vmin, self.cfg.vmax, self.cfg.num_bins
        )
        if return_type == "min":
            return q_value.min(axis=0)
        return q_value.sum(axis=0) / 2


class Ensemble(nnx.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules: list[nnx.Module]):
        super().__init__()
        self.modules = modules
        # combine_state_for_ensemble causes graph breaks
        # self.params = from_modules(*modules, as_module=True)
        # with self.params[0].data.to("meta").to_module(modules[0]):
        #     self.module = deepcopy(modules[0])
        # self._repr = str(modules[0])
        self._n = len(modules)
        
        def forward(module: nnx.Module, x: Array) -> Array:
            return module(x)
        self._forward = nnx.vmap(forward, in_axes=(0, None))

    def __len__(self):
        return self._n

    # @nnx.vmap(in_axes=(0, None))
    # def _forward(module: nnx.Module, x: Array) -> Array: # , *args, **kwargs): # TODO: removable?
    #     return module(x)
    #
    def __call__(self, x: Array) -> Array: #, *args, **kwargs): # TODO: removable?
        # #@nnx.vmap(in_axes=(0, None))
        # def _forward(module: nnx.Module, x: Array) -> Array: # , *args, **kwargs): # TODO: removable?
        #     return module(x)
        # forward = nnx.vmap(_forward, in_axes=(0, None))
        return self._forward(self.modules, x)

    # TODO: removable?
    # def __repr__(self):
    #     return f"Vectorized {len(self)}x " + self._repr


class ShiftAug(nnx.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2

    TODO: complete port

    Warnings
    --------
    This has not been fully ported to JAX/FLAX. Complete port before use!
    """

    def __init__(self, pad: int = 3):
        super().__init__()
        self.pad = pad
        self.padding = (
            (self.pad, self.pad),
        ) * 4  # pad by self.pad on each side of every dimension
        self.prng_key = None  # TODO: inject key

    def forward(self, x: Array) -> Array:
        x = x.astype(jnp.float32)
        n, _, h, w = x.shape
        assert h == w
        x = jnp.pad(x, self.padding, "edge")
        eps = 1.0 / (h + 2 * self.pad)
        arange = (
            jnp.linspace(
                start=-1.0 + eps,
                stop=1.0 - eps,
                num=h + 2 * self.pad,
                dtype=x.dtype,
            )
            .at[:h]
            .get()
        )
        arange = jnp.expand_dims(
            jnp.expand_dims(arange, 0).repeat(h, axis=1),
            axis=2,
        )
        base_grid = jnp.concat([arange, arange.transpose(1, 0)], axis=2)
        base_grid = jnp.repeat(
            jnp.expand_dims(base_grid, axis=0),
            repeats=n,
            axis=0,
        )
        shift = jax.random.randint(
            key=self.prng_key,
            shape=(n, 1, 1, 2),
            minval=0,
            maxval=2 * self.pad + 1,
            dtype=x.dtype,
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )  # TODO


class PixelPreprocess(nnx.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: Array) -> Array:
        return jnp.true_divide(x, 255.0) - 0.5


class SimNorm(nnx.Module):
    """Simplicial normalization.

    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x: Array) -> Array:
        shp = x.shape
        x = jnp.reshape(x, [*shp[:-1], -1, self.dim])
        x = nnx.softmax(x, axis=-1)
        return jnp.reshape(x, shp)

    # TODO: removable?
    # def __repr__(self):
    #     return f"SimNorm(dim={self.dim})"


class NormedLinear(nnx.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(
        self,
        *args,
        rngs: nnx.Rngs,
        dropout: float = 0.0,
        act: Callable[..., Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, rngs=rngs, **kwargs)
        self.ln = nnx.LayerNorm(
            num_features=self.out_features,
            rngs=rngs,
        )
        self.act = act if act is not None else jax.nn.mish
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout else None

    @override
    def __call__(self, x: Array) -> Array:
        x = super().__call__(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    # TODO: removable?
    # def __repr__(self):
    #     repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
    #     return (
    #         f"NormedLinear(in_features={self.in_features}, "
    #         f"out_features={self.out_features}, "
    #         f"bias={self.bias is not None}{repr_dropout}, "
    #         f"act={self.act.__class__.__name__})"
    #     )


def mlp(
    in_dim: int,
    mlp_dims: list[int] | int,
    out_dim: int,
    rngs: nnx.Rngs,
    act: Callable[..., Any] | None = None,
    dropout: float = 0.0,
) -> nnx.Sequential:
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.

    Parameters
    ----------
    in_dim
        Dimension of the input layer.
    mlp_dims
        Dimensions of the hidden layers. If just one value, there will be 1
        hidden layer with the given dimension.
    out_dim
        Dimension of the output layer.
    rngs
        Rngs.
    act
        Activation function for the output layer.
    dropout
        Dropout probability used for all hidden layers.
    """
    kernel_init = nnx.initializers.truncated_normal(0.02)
    bias_init = nnx.initializers.constant(0.0)
    if isinstance(mlp_dims, int):  # just one hidden layer
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    layers: list[nnx.Module] = []
    for i in range(len(dims) - 2):
        layers.append(
            NormedLinear(
                dims[i],
                dims[i + 1],
                rngs=rngs,
                dropout=dropout * (i == 0),
                kernel_init=kernel_init,
                bias_init=bias_init,
            )
        )
    layers.append(
        NormedLinear(
            dims[-2],
            dims[-1],
            rngs=rngs,
            act=act,
            dropout=0.0,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        if act
        else nnx.Linear(
            dims[-2],
            dims[-1],
            rngs=rngs,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
    )
    result = nnx.Sequential(*layers)
    return result


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64  # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(),
        PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg: AgentConfig, rngs: nnx.Rngs, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    for k in cfg.obs_shape.keys():
        if k == "state":
            out[k] = mlp(
                in_dim=cfg.obs_shape[k][0],
                mlp_dims=max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                out_dim=cfg.latent_dim,
                rngs=rngs,
                act=SimNorm(cfg.simnorm_dim),
            )
        elif k == "rgb":
            raise NotImplementedError(
                "Encoder for observation type rgb not fully ported to JAX/Flax."
            )
            out[k] = conv(
                cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg.simnorm_dim)
            )
        else:
            raise NotImplementedError(
                f"Encoder for observation type {k} not implemented."
            )
    return out


def api_model_conversion(target_state_dict, source_state_dict):
    """
    Converts a checkpoint from our old API to the new torch.compile compatible API.
    """
    # check whether checkpoint is already in the new format
    if "_detach_Qs_params.0.weight" in source_state_dict:
        return source_state_dict

    name_map = ["weight", "bias", "ln.weight", "ln.bias"]
    new_state_dict = dict()

    # rename keys
    for key, val in list(source_state_dict.items()):
        if key.startswith("_Qs."):
            num = key[len("_Qs.params.") :]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_Qs.params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val
            new_total_key = "_detach_Qs_params." + new_key
            new_state_dict[new_total_key] = val
        elif key.startswith("_target_Qs."):
            num = key[len("_target_Qs.params.") :]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_target_Qs_params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val

    # add batch_size and device from target_state_dict to new_state_dict
    for prefix in ("_Qs.", "_detach_Qs_", "_target_Qs_"):
        for key in ("__batch_size", "__device"):
            new_key = prefix + "params." + key
            new_state_dict[new_key] = target_state_dict[new_key]

    # check that every key in new_state_dict is in target_state_dict
    for key in new_state_dict:
        assert key in target_state_dict, f"key {key} not in target_state_dict"
    # check that all Qs keys in target_state_dict are in new_state_dict
    for key in target_state_dict.keys():
        if "Qs" in key:
            assert key in new_state_dict, f"key {key} not in new_state_dict"
    # check that source_state_dict contains no Qs keys
    for key in source_state_dict.keys():
        assert "Qs" not in key, f"key {key} contains 'Qs'"

    # copy log_std_min and log_std_max from target_state_dict to new_state_dict
    new_state_dict["log_std_min"] = target_state_dict["log_std_min"]
    new_state_dict["log_std_dif"] = target_state_dict["log_std_dif"]
    new_state_dict["_action_masks"] = target_state_dict["_action_masks"]

    # copy new_state_dict to source_state_dict
    source_state_dict.update(new_state_dict)

    return source_state_dict


# TODO: removable?
# TODO: initialize ParameterList? Where is this anyway?
# def weight_init(m):
#     """Custom weight initialization for TD-MPC2."""
#     elif isinstance(m, nn.ParameterList):
#         for i, p in enumerate(m):
#             if p.dim() == 3:  # Linear
#                 nn.init.trunc_normal_(p, std=0.02)  # Weight
#                 nn.init.constant_(m[i + 1], 0)  # Bias


def complete_config(
    env: gym.Env,
    agent_cfg: AgentConfig,
    training_cfg: TrainingConfig,
) -> (AgentConfig, TrainingConfig):
    """Fill in some configuration values that are based on others.

    TODO: Document which config fields are set here and which other fields they depend on.

    Parameters
    ----------
    env : gym.Env
    agent_cfg : AgentConfig
        The potentially incomplete agent-specific configuration.
    training_cfg : TrainingConfig
        The potentially incomplete training-specific configuration.

    Returns
    -------
        Both completed parts of the configuration.

    """
    try:  # Dict
        agent_cfg.obs_shape = {
            k: v.shape for k, v in env.observation_space.spaces.items()
        }
    except:  # Box
        agent_cfg.obs_shape = {agent_cfg.obs: env.observation_space.shape}
    # Bin size for discrete regression
    agent_cfg.bin_size = (agent_cfg.vmax - agent_cfg.vmin) / (
        agent_cfg.num_bins - 1
    )
    agent_cfg.action_dim = env.action_space.shape[0]
    agent_cfg.episode_length = env.spec.max_episode_steps
    if training_cfg.seed_steps is None:
        training_cfg.seed_steps = max(1000, 5 * agent_cfg.episode_length)
    # Heuristic for large action spaces
    agent_cfg.iterations += 2 * int(agent_cfg.action_dim >= 20)
    return agent_cfg, training_cfg


def train_tdmpc2(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    agent_cfg: AgentConfig,
    training_cfg: TrainingConfig,
    rng_seed: int = 1,
    logger: LoggerBase | None = None,
    timer: Timer = Timer(),
) -> TDMPC2:
    """TD-MPC2 from [1]_.

    Note that some parts are not described in [1]_ but only in [2]_.

    Parameters
    ----------
    env : gymnasium.Env
        Training environment.
    agent_cfg : AgentConfig
        Agent-specific configuration
    training_cfg : TrainingConfig
        Training-specific configuration
    rng_seed : int
        Seed used for all RNGs.
    logger : LoggerBase, optional
        Experiment logger.
    timer : Timer
        Timer to profile select parts of the training process, logs to logger.

    See Also
    --------
    make_agent_config :
        Create the agent-specific configuration; with doc.
    make_training_config :
        Create the training-specific configuration; with doc.

    References
    ----------
    .. [1] Hansen, N., Su, H., & Wang, X. (2024). TD-MPC2: Scalable, Robust
       World Models for Continuous Control (arXiv:2310.16828).
       https://doi.org/10.48550/arXiv.2310.16828
    .. [2] Hansen, N., Wang, X., & Su, H. (2022). Temporal Difference Learning
       for Model Predictive Control (arXiv:2203.04955).
       https://doi.org/10.48550/arXiv.2203.04955
    """
    # Check parameters, compute defaults, and create configuration object
    assert training_cfg.steps > 0, "Must train for at least 1 step."

    agent_cfg, training_cfg = complete_config(env, agent_cfg, training_cfg)

    gym.logger.min_level = 40
    env = DefaultSuccessInfoWrapper(env)

    full_cfg = FullTrainingConfig(*agent_cfg, *training_cfg)

    trainer = OnlineTrainer(
        cfg=full_cfg,
        env=env,
        agent=TDMPC2.from_config(agent_cfg, rng_seed),
        logger=logger,
        timer=timer,
    )
    trainer.train()
    print("\nTraining completed successfully")
    return trainer.agent
