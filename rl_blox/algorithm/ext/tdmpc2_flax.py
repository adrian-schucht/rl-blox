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

from recordclass import recordclass, dataobject, asdict

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
import time
from collections import defaultdict, namedtuple
from collections.abc import Callable
from functools import partial
from typing import Any, override, Literal

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import lax, Array
from jax.typing import ArrayLike
from tqdm.rich import trange

from rl_blox.logging.logger import LoggerBase
from rl_blox.logging.timer import Timer
from rl_blox.blox.losses import mse_loss
from rl_blox.blox.replay_buffer import SubtrajectoryReplayBuffer
from rl_blox.blox.target_net import soft_target_net_update

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


class TDMPC2AgentState(nnx.Module):
    @staticmethod
    def create_from(cfg: AgentConfig, rngs: nnx.Rngs):
        return TDMPC2AgentState(
            model=WorldModel(cfg, rngs),
            pi=create_tdmpc2_pi(cfg, rngs),
            previous_mean=jnp.zeros(shape=(cfg.horizon, cfg.action_dim)),
        )

    def __init__(self, model: WorldModel, pi: nnx.Module, previous_mean: Array):
        self.model = model
        self.pi = pi
        self.previous_mean = nnx.Variable(previous_mean)


class TDMPC2TrainState:
    def __init__(
        self,
        agent_state: TDMPC2AgentState,
        model_optimizer: nnx.Optimizer,
        pi_optimizer: nnx.Optimizer,
    ):
        self.agent_state = agent_state
        self.model_optimizer = model_optimizer
        self.pi_optimizer = pi_optimizer


AgentConfig = recordclass(
    "AgentConfig",
    [
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
        "discount",
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
        "log_std_dif",
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
        "discount",
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
        "log_std_dif",
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
        log_std_dif=None,  # Set during training
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
        bin_size=None,  # Set during training
        action_dim=None,  # Set during training
        episode_length=None,  # Set during training
        obs_shape=None,  # Set during training
        discount=None,  # Set during training
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
    x = jnp.clip(symlog(x), vmin, vmax).squeeze(axis=-1)
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
    dreg_bins = jnp.linspace(vmin, vmax, num_bins, dtype=x.dtype)
    x = nnx.softmax(x, axis=-1)
    x = jnp.sum(x * dreg_bins, axis=-1, keepdims=True)
    return symexp(x)


def gumbel_softmax_sample(
    p: ArrayLike,
    rngs: nnx.Rngs,
    temperature: float = 1.0,
    dim: int = 0,
):
    logits = jnp.log(p)
    # Generate Gumbel noise
    gumbels = -jnp.log(
        rngs.exponential(
            shape=logits.shape,
            dtype=logits.dtype,
        )
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = nnx.softmax(gumbels, axis=dim)
    return jnp.argmax(y_soft, axis=-1)


class RunningScale(nnx.Module):
    """Running trimmed scale estimator."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.value = nnx.Variable(jnp.array(1.0))
        self.updates = jnp.array(0, dtype=int)
        self.mean = jnp.array(0)
        self.std = jnp.array(0)
        self._percentiles = nnx.Variable(jnp.array([5, 95]))

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
        in_sorted = jnp.sort(x, axis=0)
        floored, ceiled, weight_floored, weight_ceiled = self._positions(
            x.shape[0]
        )
        d0 = in_sorted[floored] * weight_floored
        d1 = in_sorted[ceiled] * weight_ceiled
        return jnp.reshape(d0 + d1, (-1,) + x_shape[1:]).astype(x_dtype)

    def update(self, x: Array):
        x = x.squeeze()
        percentiles = self._percentile(x)  # NOTE: previously detach()
        value = jnp.clip(percentiles[1] - percentiles[0], min=1.0)
        self.value = jnp.interp(
            x=self.cfg.tau,
            xp=jnp.array([0.0, 1.0]),
            fp=jnp.array([self.value, value]),
        )
        self.updates = self.updates + 1
        self.mean = jnp.mean(x)
        self.std = jnp.std(x)

    def __call__(self, x: Array, update=False):
        if update:
            self.update(x)
        return x / self.value


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


def _train(
    cfg: FullTrainingConfig,
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    train_state: TDMPC2TrainState,
    rngs: nnx.Rngs,
    np_rng: np.random.Generator,
    logger: LoggerBase | None = None,
    timer: Timer = Timer(),
) -> TDMPC2AgentState:
    """Train a TD-MPC2 agent."""
    buffer = SubtrajectoryReplayBuffer(
        buffer_size=min(cfg.buffer_size, cfg.steps),
        horizon=cfg.horizon,
    )
    scale = RunningScale(cfg)
    step = 0
    ep_idx = 0
    start_time = time.time()
    transitions_in_episode = []
    done = True
    steps_in_episode = 0
    episode_reward = 0.0
    agent_state = train_state.agent_state
    model = agent_state.model
    pi = agent_state.pi
    previous_mean = agent_state.previous_mean
    model_optim = train_state.model_optimizer
    pi_optim = train_state.pi_optimizer

    progress = trange(step, cfg.steps, disable=not cfg.progress_bar)

    timer.start("training")
    timer.start("seed_acquisition")
    for step in np.arange(step, cfg.steps + 1):
        # Reset environment
        if done:
            if step > 0:
                episode_success = info["success"]
                if logger is not None:
                    logger.record_stat("return", value=episode_reward)
                    logger.record_stat("success", value=episode_success)
                    logger.stop_episode(steps_in_episode)

                steps_in_episode = 0
                episode_reward = 0.0
                ep_idx += 1

            if logger is not None:
                logger.start_new_episode()

            obs, _ = env.reset()

        # Collect experience
        if step > cfg.seed_steps:
            timer.start("agent_act")
            t0 = steps_in_episode == 0
            action, previous_mean = act(
                model=model,
                pi=pi,
                obs=obs,
                previous_mean=previous_mean,
                rngs=rngs,
                cfg=cfg,
                t0=t0,
            )
            timer.stop("agent_act")
        else:
            timer.start("env_sample_action_space")
            action = env.action_space.sample()
            timer.stop("env_sample_action_space")
        prev_obs = obs
        timer.start("env_step")
        obs, reward, termination, truncation, info = env.step(action)
        timer.stop("env_step")
        done = termination or truncation
        _ = buffer.add_sample(
            observation=prev_obs,
            action=action,
            reward=reward,
            next_observation=obs,
            terminated=termination,
            truncated=truncation,
        )
        steps_in_episode += 1
        episode_reward += float(reward)

        # Update agent
        if step >= cfg.seed_steps:
            if step == cfg.seed_steps:
                num_updates = cfg.seed_steps
                timer.stop("seed_acquisition")
                print("Pretraining agent on seed data...")
            else:
                num_updates = 1
            for i in range(num_updates):
                timer.start("agent_update")
                metrics = update(
                    model=model,
                    pi=pi,
                    model_optim=model_optim,
                    pi_optim=pi_optim,
                    scale=scale,
                    buffer=buffer,
                    rngs=rngs,
                    np_rng=np_rng,
                    cfg=cfg,
                )
                timer.stop("agent_update")
                progress.update()  # 1 update = 1 step, just not necessarily synchronously
                if i == num_updates - 1 and logger is not None:
                    for k, v in metrics.items():
                        logger.record_stat(k, v)
            if logger is not None:
                logger.record_epoch(
                    key="result",
                    value=TDMPC2AgentState(
                        model=model,
                        pi=pi,
                        previous_mean=previous_mean,
                    ),
                    step=step,
                )

    for i in range(1001):
        if i == 1:
            timer.start("acting")
        test_obs = jnp.array(env.observation_space.sample())
        act(
            model=model,
            pi=pi,
            obs=test_obs,
            previous_mean=previous_mean,
            rngs=rngs,
            cfg=cfg,
            t0=False,
        )
    timer.stop("acting")

    # End last (potentially partial) episode
    if logger is not None:
        timer.stop("training")
        timer.log(logger)
        logger.stop_episode(steps_in_episode)

    return TDMPC2AgentState(
        model=model,
        pi=pi,
        previous_mean=previous_mean,
    )


def create_tdmpc2_pi(cfg: AgentConfig, rngs: nnx.Rngs):
    return mlp(
        in_dim=cfg.latent_dim,
        mlp_dims=2 * [cfg.mlp_dim],
        out_dim=2 * cfg.action_dim,
        rngs=rngs,
    )


def create_tdmpc2_train_state(cfg: AgentConfig, seed: int = 0):
    rngs = nnx.Rngs(seed)
    agent_state = TDMPC2AgentState.create_from(cfg, rngs)
    labeled_state = nnx.State(
        {
            "_encoder": "encoder",
            "_dynamics": "default",
            "_reward": "default",
            "_Qs": "default",
            "_target_Qs": "off",
        }
    )
    model_optimizer = nnx.Optimizer(
        agent_state.model,
        optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip_norm),
            optax.partition(
                {
                    "encoder": optax.adam(
                        learning_rate=cfg.lr * cfg.enc_lr_scale
                    ),
                    "default": optax.adam(learning_rate=cfg.lr),
                    "off": optax.identity(),
                },
                labeled_state,
            ),
        ),
        wrt=nnx.Param,
    )
    pi_optimizer = nnx.Optimizer(
        agent_state.pi,
        optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip_norm),
            optax.adam(
                learning_rate=cfg.lr,
                eps=1e-5,
            ),
        ),
        wrt=nnx.Param,
    )
    agent_state.model.eval()
    agent_state.pi.eval()
    return TDMPC2TrainState(
        agent_state=agent_state,
        model_optimizer=model_optimizer,
        pi_optimizer=pi_optimizer,
    )


def _make_result(
    model: WorldModel,
    pi: nnx.Module,
    previous_mean: Array,
):
    return namedtuple(
        "TDMPC2Result",
        [
            "model",
            "pi",
            "previous_mean",
        ],
    )(
        model,
        pi,
        previous_mean,
    )


@partial(nnx.jit, static_argnames=["cfg", "eval_mode"])
def act(
    model: WorldModel,
    pi: nnx.Module,
    obs: ArrayLike,
    previous_mean: Array | None,
    rngs: nnx.Rngs,
    cfg: AgentConfig,
    t0: bool = False,
    eval_mode: bool = False,
) -> tuple[Array, Array]:
    """Select an action by planning in the latent space of the world model.

    Parameters
    ----------
    obs
        Observation from the environment.
    t0
        Whether this is the first observation in the episode.
    eval_mode
        Whether to use the mean of the action distribution.

    Returns
    -------
    torch.Tensor
        Action to take in the environment.
    """
    obs = jnp.expand_dims(obs, axis=0)
    if cfg.mpc:
        return _plan(
            model=model,
            pi=pi,
            previous_mean=previous_mean,
            obs=obs,
            rngs=rngs,
            cfg=cfg,
            t0=t0,
            eval_mode=eval_mode,
        )
    z = model.encode(obs)
    action, info = sample_pi(pi, z, rngs, cfg)
    if eval_mode:
        action = info["mean"]
    return namedtuple(
        "PlanningResult",
        [
            "action",
            "mean",
        ],
    )(
        action.at[0].get(),
        info["mean"],
    )


def _estimate_value(
    cfg: AgentConfig,
    model: WorldModel,
    pi: nnx.Module,
    z: Array,
    actions: Array,
    rngs: nnx.Rngs,
):
    """Estimate value of a trajectory starting at latent state z and
    executing given actions.

    (ll. 5-9, Algorithm 1, TD-MPC (inference), [2]_)
    """
    G, discount = 0, 1
    for t in range(cfg.horizon):
        reward = two_hot_inv(
            model.reward_fwd(z, actions.at[t].get()),
            cfg.vmin,
            cfg.vmax,
            cfg.num_bins,
        )
        z = model.next(z, actions[t])
        G = G + discount * reward
        discount_update = cfg.discount
        discount = discount * discount_update
    action, _ = sample_pi(pi, z, rngs, cfg)
    return G + discount * model.Q(z, action, rngs=rngs, return_type="avg")


@partial(nnx.jit, static_argnames=["cfg", "eval_mode"])
def _plan(
    model: WorldModel,
    pi: nnx.Module,
    previous_mean: Array | None,
    obs: Array,
    rngs: nnx.Rngs,
    cfg: AgentConfig,
    t0: bool = False,
    eval_mode: bool = False,
) -> tuple[Array, Array]:
    """Plan a sequence of actions using the learned world model.

    Parameters
    ----------
    obs
        State whose latent representation to plan from.
    t0
        Whether this is the first observation in the episode.
    eval_mode
        Whether to use the mean of the action distribution.

    Returns
    -------
    Array
        Action to take in the environment.
    """
    if previous_mean is None:
        previous_mean = jnp.zeros(shape=(cfg.horizon, cfg.action_dim))

    # Sample policy trajectories.
    # (l. 4, Algorithm 1, TD-MPC (inference), [2]_)
    z = model.encode(obs)
    if cfg.num_pi_trajs > 0:
        pi_actions = jnp.empty(
            shape=(
                cfg.horizon,
                cfg.num_pi_trajs,
                cfg.action_dim,
            )
        )
        _z = jnp.repeat(
            z,
            repeats=cfg.num_pi_trajs,
            axis=0,
        )
        for t in range(cfg.horizon - 1):
            action, _ = sample_pi(pi, _z, rngs, cfg)
            pi_actions = pi_actions.at[t].set(action)
            _z = model.next(_z, pi_actions[t])
        action, _ = sample_pi(pi, _z, rngs, cfg)
        pi_actions = pi_actions.at[-1].set(action)

    # Initialize state and parameters
    z = jnp.repeat(
        z,
        repeats=cfg.num_samples,
        axis=0,
    )
    mean = jnp.zeros((cfg.horizon, cfg.action_dim))
    std = jnp.full(
        (cfg.horizon, cfg.action_dim),
        fill_value=cfg.max_std,
        dtype=jnp.float_,
    )
    mean = mean.at[:-1].set(
        lax.cond(
            t0,
            lambda a: a[0],
            lambda a: a[1],
            (mean.at[:-1].get(), previous_mean.at[1:].get()),
        )
    )
    actions = jnp.empty(
        shape=(
            cfg.horizon,
            cfg.num_samples,
            cfg.action_dim,
        ),
    )
    if cfg.num_pi_trajs > 0:
        actions = actions.at[:, : cfg.num_pi_trajs].set(pi_actions)

    # Iterate MPPI
    # (ll. 2-10, Algorithm 1, TD-MPC (inference), [2]_)
    for _ in range(cfg.iterations):
        # Sample MPPI actions
        # (l. 3, Algorithm 1, TD-MPC (inference), [2]_)
        r = rngs.normal(
            shape=(
                cfg.horizon,
                cfg.num_samples - cfg.num_pi_trajs,
                cfg.action_dim,
            )
        )
        actions_sample = jnp.expand_dims(mean, 1) + jnp.expand_dims(std, 1) * r
        actions_sample = jnp.clip(actions_sample, -1.0, 1.0)
        actions = actions.at[:, cfg.num_pi_trajs :].set(actions_sample)

        # Compute elite actions
        value = _estimate_value(cfg, model, pi, z, actions, rngs)
        _, elite_idxs = jax.lax.top_k(
            value.squeeze(1),
            cfg.num_elites,
        )
        elite_value = value.at[elite_idxs].get()
        elite_actions = actions.at[:, elite_idxs].get()

        # Update parameters
        # (l. 10, Algorithm 1, TD-MPC (inference), [2]_)
        max_value = jnp.max(elite_value, axis=0)
        score = jnp.exp(cfg.temperature * (elite_value - max_value))
        score = score / jnp.sum(score, axis=0)
        mean = jnp.sum(
            jnp.expand_dims(score, 0) * elite_actions,
            axis=1,
        ) / (jnp.sum(score, 0) + 1e-9)
        std = jnp.sqrt(
            jnp.sum(
                (
                    jnp.expand_dims(score, 0)
                    * (elite_actions - jnp.expand_dims(mean, 1)) ** 2
                ),
                axis=1,
            )
            / (jnp.sum(score, 0) + 1e-9)
        )
        std = jnp.clip(std, cfg.min_std, cfg.max_std)

    # Select action
    # (l. 11, Algorithm 1, TD-MPC (inference), [2]_)
    rand_idx = gumbel_softmax_sample(
        score.squeeze(1),
        rngs,
    )
    actions = jnp.squeeze(jnp.take(elite_actions, rand_idx, axis=1), axis=1)
    a, std = actions.at[0].get(), std.at[0].get()
    if not eval_mode:
        a = a + std * rngs.normal(cfg.action_dim)
    else:
        a = jnp.expand_dims(a, axis=0)
    return namedtuple(
        "PlanningResult",
        [
            "action",
            "mean",
        ],
    )(
        jnp.clip(a, -1.0, 1.0),
        mean,
    )


def _pi_loss(
    pi: nnx.Module,
    model: WorldModel,
    scale: RunningScale,
    zs: Array,
    rngs: nnx.Rngs,
    cfg: AgentConfig,
) -> tuple[Array, dict[str, Array]]:
    """Calculate the loss of the policy on a sequence of latent states.

    (Equation 4, Policy objective, [2]_)

    Parameters
    ----------
    zs
        Sequence of latent states.

    Returns
    -------
    tuple[Array, dict[str, Array]]
        A pair of (1) the policy loss and (2) the info dict from
        WorldModel.pi().
    """
    action, info = sample_pi(pi, zs, rngs, cfg)
    qs = model.Q(zs, action, rngs=rngs, return_type="avg")
    scale.update(qs.at[0].get())
    qs = scale(qs)

    # Loss is a weighted sum of Q-values
    # (rho is lambda in Equation (4), [2]_)
    rho = jnp.pow(cfg.rho, jnp.arange(len(qs)))
    pi_loss = (
        -(cfg.entropy_coef * info["scaled_entropy"] + qs).mean(axis=(1, 2))
        * rho
    ).mean()
    return pi_loss, info


def update_pi(
    pi: nnx.Module,
    model: WorldModel,
    pi_optim: nnx.Optimizer,
    scale: RunningScale,
    zs: Array,
    rngs: nnx.Rngs,
    cfg: AgentConfig,
):
    """Update the policy using a sequence of latent states.

    Parameters
    ----------
    zs
        Sequence of latent states.

    Returns
    -------
    float
        Loss of the policy update.

    See Also
    --------
    _pi_loss
    """
    (pi_loss, info), pi_loss_grads = nnx.value_and_grad(
        _pi_loss,
        argnums=0,
        has_aux=True,
    )(pi, model, scale, zs, rngs, cfg)
    pi_grad_norm = optax.tree_utils.tree_norm(pi_loss_grads, ord=2)
    pi_optim.update(pi, pi_loss_grads)

    info = {
        "policy loss": pi_loss,
        "policy grad norm": pi_grad_norm,
        "policy entropy": info["entropy"],
        "policy scaled entropy": info["scaled_entropy"],
        "policy scale": scale.value,
        "policy scale updates": scale.updates,  # TODO: remove
        "policy scale last mean": scale.mean,  # TODO: remove
        "policy scale last std": scale.std,  # TODO: remove
    }
    return info


def _td_target(
    cfg: AgentConfig,
    model: WorldModel,
    pi: nnx.Module,
    next_z: Array,
    reward: Array,
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
    rngs
        Rngs for invoking the policy.

    Returns
    -------
    Array
        TD-target.
    """
    action, _ = sample_pi(pi, next_z, rngs, cfg)
    return reward + cfg.discount * model.Q(
        next_z, action, rngs=rngs, return_type="min", target=True
    )


def _model_loss(
    model: WorldModel,
    pi: nnx.Module,
    obs: Array,
    action: Array,
    reward: Array,
    rngs: nnx.Rngs,
    cfg: AgentConfig,
) -> tuple[Array, dict[str, Array]]:
    # Compute targets
    next_z = lax.stop_gradient(model.encode(obs[1:]))
    td_targets = lax.stop_gradient(
        _td_target(cfg, model, pi, next_z, reward, rngs)
    )

    # Prepare for update
    model.train()

    # Latent rollout
    zs = jnp.zeros(
        (
            cfg.horizon + 1,
            cfg.batch_size,
            cfg.latent_dim,
        )
    )
    z = model.encode(obs.at[0].get())
    zs = zs.at[0].set(z)
    consistency_loss = 0
    for t, (_action, _next_z) in enumerate(
        zip(
            jnp.unstack(action, axis=0),
            jnp.unstack(next_z, axis=0),
            strict=False,
        )
    ):
        z = model.next(z, _action)
        consistency_loss = consistency_loss + mse_loss(z, _next_z) * cfg.rho**t
        zs = zs.at[t + 1].set(z)

    # Predictions
    _zs = zs.at[:-1].get()
    qs = model.Q(_zs, action, return_type="all")
    reward_preds = model.reward_fwd(_zs, action)

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
                cfg.vmin,
                cfg.vmax,
                cfg.bin_size,
                cfg.num_bins,
            ).mean()
            * cfg.rho**t
        )
        for _, qs_unbind_unbind in enumerate(jnp.unstack(qs_unbind, axis=0)):
            value_loss = (
                value_loss
                + soft_ce(
                    qs_unbind_unbind,
                    td_targets_unbind,
                    cfg.vmin,
                    cfg.vmax,
                    cfg.bin_size,
                    cfg.num_bins,
                ).mean()
                * cfg.rho**t
            )

    consistency_loss = consistency_loss / cfg.horizon
    reward_loss = reward_loss / cfg.horizon
    value_loss = value_loss / (cfg.horizon * cfg.num_q)
    total_loss = (
        cfg.consistency_coef * consistency_loss
        + cfg.reward_coef * reward_loss
        + cfg.value_coef * value_loss
    )

    info = {
        "consistency loss": consistency_loss,
        "reward loss": reward_loss,
        "q loss": value_loss,
        "total loss": total_loss,
    }

    return total_loss, (info, zs)


@partial(nnx.jit, static_argnames=["cfg"])
def _update(
    model: WorldModel,
    pi: nnx.Module,
    model_optim: nnx.Optimizer,
    pi_optim: nnx.Optimizer,
    scale: RunningScale,
    obs: Array,
    action: Array,
    reward: Array,
    rngs: nnx.Rngs,
    cfg: AgentConfig,
):
    # Update model
    (model_loss, (model_info, zs)), model_loss_grads = nnx.value_and_grad(
        _model_loss,
        argnums=0,
        has_aux=True,
    )(model, pi, obs, action, reward, rngs, cfg)
    model_grad_norm = optax.tree_utils.tree_norm(model_loss_grads, ord=2)
    model_optim.update(model, model_loss_grads)

    # Update policy
    pi_info = update_pi(pi, model, pi_optim, scale, zs, rngs, cfg)

    # Update target Q-functions
    soft_target_net_update(
        net=model.Qs,
        target_net=model.target_Qs,
        tau=cfg.tau,
    )

    # Return training statistics
    model.eval()
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
    obs = jnp.swapaxes(obs, 0, 1)
    action = jnp.swapaxes(action, 0, 1)
    reward = jnp.expand_dims(
        jnp.swapaxes(reward, 0, 1),
        axis=-1,
    )
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


def update(
    model: WorldModel,
    pi: nnx.Module,
    model_optim: nnx.Optimizer,
    pi_optim: nnx.Optimizer,
    scale: RunningScale,
    buffer: SubtrajectoryReplayBuffer,
    rngs: nnx.Rngs,
    np_rng: np.random.Generator,
    cfg: AgentConfig,
):
    """
    Main update function. Corresponds to one iteration of model learning.

    Args:
            buffer (common.buffer.Buffer): Replay buffer.

    Returns:
            dict: Dictionary of training statistics.
    """
    batch = buffer.sample_batch(
        batch_size=cfg.batch_size,
        horizon=cfg.horizon,
        include_intermediate=True,
        rng=np_rng,
    )
    prepared_batch = _prepare_batch(batch)
    obs, action, reward = prepared_batch
    return _update(
        model=model,
        pi=pi,
        model_optim=model_optim,
        pi_optim=pi_optim,
        scale=scale,
        obs=obs,
        action=action,
        reward=reward,
        rngs=rngs,
        cfg=cfg,
    )


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
        self.encoder = enc(cfg, rngs)
        self.dynamics = mlp(
            in_dim=cfg.latent_dim + cfg.action_dim,
            mlp_dims=2 * [cfg.mlp_dim],
            out_dim=cfg.latent_dim,
            rngs=rngs,
            act=SimNorm(cfg.simnorm_dim),
        )
        self.reward = mlp(
            in_dim=cfg.latent_dim + cfg.action_dim,
            mlp_dims=2 * [cfg.mlp_dim],
            out_dim=max(cfg.num_bins, 1),
            rngs=rngs,
            last_layer_inits_to_zero=True,
        )

        def make_single_Q(rngs: nnx.Rngs):
            return mlp(
                in_dim=cfg.latent_dim + cfg.action_dim,
                mlp_dims=2 * [cfg.mlp_dim],
                out_dim=max(cfg.num_bins, 1),
                rngs=rngs,
                dropout=cfg.dropout,
                last_layer_inits_to_zero=True,
            )

        self.Qs = Ensemble(make_single_Q, n=self.cfg.num_q, rngs=rngs)
        self.target_Qs = nnx.clone(self.Qs)

    @override
    def train(self, **attributes):
        """Overriding `train` method to keep target Q-networks in eval mode."""
        super().train(**attributes)
        self.target_Qs.eval()

    def encode(self, obs: Array):
        """Encodes an observation into its latent representation.

        This implementation assumes a single state-based observation.
        """
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return jnp.stack([self.encoder[self.cfg.obs](o) for o in obs])
        return self.encoder[self.cfg.obs](obs)

    def next(self, z: ArrayLike, a: ArrayLike) -> Array:
        """Predicts the next latent state given the current latent state
        and action.

        Latent dynamics. In the paper: d(z,a,e).
        """
        z = jnp.concat([z, a], axis=-1)
        return self.dynamics(z)

    def reward_fwd(self, z: ArrayLike, a: ArrayLike) -> Array:
        """Predicts instantaneous (single-step) reward.

        Reward. In the paper: R(z,a,e).
        """
        z = jnp.concat([z, a], axis=-1)
        return self.reward(z)

    def Q(
        self,
        z: Array,
        a: Array,
        return_type: Literal["min", "avg", "all"] = "min",
        rngs: nnx.Rngs | None = None,
        target: bool = False,
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
            qnet = self.target_Qs
        else:
            qnet = self.Qs
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


def sample_pi(
    pi: nnx.Module, z: ArrayLike, rngs: nnx.Rngs, cfg: AgentConfig
) -> tuple[Array, dict[str, Array]]:
    """Samples an action from the policy prior.

    The policy prior is a Gaussian distribution with
    mean and (log) std predicted by a neural network.
    """
    # Gaussian policy prior
    mean, log_std = jnp.split(pi(z), 2, axis=-1)
    log_std = safe_log_std(
        log_std, jnp.array(cfg.log_std_min), jnp.array(cfg.log_std_dif)
    )
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


class Ensemble(nnx.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(
        self,
        make_module: Callable[[nnx.Rngs], nnx.Module],
        n: int,
        rngs: nnx.Rngs,
    ):
        @nnx.split_rngs(splits=n)
        @nnx.vmap
        def _make_module(rngs: nnx.Rngs):
            return make_module(rngs)

        self.modules = _make_module(rngs)
        self._n = n

        def forward(module: nnx.Module, x: Array) -> Array:
            return module(x)

        self._forward = nnx.vmap(forward, in_axes=(0, None))

    def __call__(self, x: Array) -> Array:
        return self._forward(self.modules, x)


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


def mlp(
    in_dim: int,
    mlp_dims: list[int] | int,
    out_dim: int,
    rngs: nnx.Rngs,
    act: Callable[..., Any] | None = None,
    dropout: float = 0.0,
    last_layer_inits_to_zero: bool = False,
) -> nnx.Sequential:
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    Weights are initialized according to a truncated normal distribution,
    unless last_layer_inits_to_zero. Biases are always initialized to 0.

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
    last_layer_inits_to_zero
        If True, the weights of the last layer are initialized to zero.

    """
    std = 0.02
    kernel_init = nnx.initializers.truncated_normal(
        stddev=std,
        lower=-2.0 / std,
        upper=2.0 / std,
    )
    last_layer_kernel_init = (
        nnx.initializers.constant(0.0)
        if last_layer_inits_to_zero
        else kernel_init
    )
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
            kernel_init=last_layer_kernel_init,
            bias_init=bias_init,
        )
        if act
        else nnx.Linear(
            dims[-2],
            dims[-1],
            rngs=rngs,
            kernel_init=last_layer_kernel_init,
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


def discount_heuristic(
    episode_length: int,
    discount_denom: int,
    discount_min: float,
    discount_max: float,
) -> float:
    """Returns the discount factor for a given episode length.

    Simple heuristic that scales discount linearly with episode length.
    Default values should work well for most tasks, but can be changed
    as needed.

    Args
    ----
    episode_length
        Length of the episode. Assumes episodes are of fixed length.

    discount_denom
        Denominator in the heuristic for the discount factor, which changes
        in response to the episode length, within the bounds of
        [discount_min,discount_max].

    discount_min
        Minimum value for the discount factor.

    discount_max
        Maximum value for the discount factor.

    Returns
    -------
    float
        Discount factor for the task.
    """
    frac = episode_length / discount_denom
    return min(
        max(
            (frac - 1) / (frac),
            discount_min,
        ),
        discount_max,
    )


def complete_config(
    env: gym.Env,
    agent_cfg: AgentConfig,
    training_cfg: TrainingConfig,
) -> tuple[AgentConfig, TrainingConfig]:
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
    agent_cfg.discount = discount_heuristic(
        episode_length=agent_cfg.episode_length,
        discount_denom=agent_cfg.discount_denom,
        discount_min=agent_cfg.discount_min,
        discount_max=agent_cfg.discount_max,
    )
    agent_cfg.log_std_dif = agent_cfg.log_std_max - agent_cfg.log_std_min
    return agent_cfg, training_cfg


def train_tdmpc2(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    agent_cfg: AgentConfig,
    training_cfg: TrainingConfig,
    seed: int = 1,
    logger: LoggerBase | None = None,
    timer: Timer = Timer(),
) -> TDMPC2AgentState:
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

    gym.logger.min_level = 40
    env = DefaultSuccessInfoWrapper(env)

    agent_cfg, training_cfg = complete_config(env, agent_cfg, training_cfg)
    full_cfg = FullTrainingConfig(*agent_cfg, *training_cfg)
    train_state = create_tdmpc2_train_state(agent_cfg, seed)

    return _train(
        cfg=full_cfg,
        env=env,
        train_state=train_state,
        rngs=nnx.Rngs(seed),
        np_rng=np.random.default_rng(seed),
        logger=logger,
        timer=timer,
    )
