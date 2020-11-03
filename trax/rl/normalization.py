# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Normalization helpers."""

import gin
import numpy as np

from trax import fastmath
from trax import layers as tl


def running_mean_init(shape, fill_value=0):
    """
    Initialize the mean of a set.

    Args:
        shape: (int): write your description
        fill_value: (todo): write your description
    """
  return (np.full(shape, fill_value), np.array(0))


def running_mean_update(x, state):
    """
    Update the mean of the mean

    Args:
        x: (int): write your description
        state: (str): write your description
    """
  (mean, n) = state
  mean = n.astype(np.float32) / (n + 1) * mean + x / (n + 1)
  return (mean, n + 1)


def running_mean_get_mean(state):
    """
    Return the mean of a state.

    Args:
        state: (str): write your description
    """
  (mean, _) = state
  return mean


def running_mean_get_count(state):
    """
    Return the number of running jobs

    Args:
        state: (str): write your description
    """
  (_, count) = state
  return count


def running_mean_and_variance_init(shape):
    """
    Initialize a state.

    Args:
        shape: (int): write your description
    """
  mean_state = running_mean_init(shape, fill_value=0.0)
  var_state = running_mean_init(shape, fill_value=1.0)
  return (mean_state, var_state)


def running_mean_and_variance_update(x, state):
    """
    Compute the mean and variance.

    Args:
        x: (array): write your description
        state: (todo): write your description
    """
  (mean_state, var_state) = state
  old_mean = running_mean_get_mean(mean_state)
  mean_state = running_mean_update(x, mean_state)
  new_mean = running_mean_get_mean(mean_state)

  var_state = running_mean_update((x - new_mean) * (x - old_mean), var_state)

  return (mean_state, var_state)


def running_mean_and_variance_get_mean(state):
    """
    Return the mean and mean and mean mean.

    Args:
        state: (todo): write your description
    """
  (mean_state, _) = state
  return running_mean_get_mean(mean_state)


def running_mean_and_variance_get_count(state):
    """
    Return the number of running running running running running running running state.

    Args:
        state: (todo): write your description
    """
  (mean_state, _) = state
  return running_mean_get_count(mean_state)


def running_mean_and_variance_get_variance(state):
    """
    Return the mean mean of a var.

    Args:
        state: (todo): write your description
    """
  (_, var_state) = state
  return running_mean_get_mean(var_state)


@gin.configurable(blacklist=['mode'])
class Normalize(tl.Layer):
  """Numerically stable normalization layer."""

  def __init__(self, sample_limit=float('+inf'), epsilon=1e-5, mode='train'):
      """
      Initialize the sample.

      Args:
          self: (todo): write your description
          sample_limit: (int): write your description
          float: (todo): write your description
          epsilon: (float): write your description
          mode: (todo): write your description
      """
    super().__init__()
    self._sample_limit = sample_limit
    self._epsilon = epsilon
    self._mode = mode

  def init_weights_and_state(self, input_signature):
      """
      Initialize weights and weights.

      Args:
          self: (todo): write your description
          input_signature: (bool): write your description
      """
    self.state = running_mean_and_variance_init(input_signature.shape[2:])

  def forward(self, inputs):
      """
      Perform forward pass

      Args:
          self: (todo): write your description
          inputs: (todo): write your description
      """
    state = self.state
    observations = inputs
    if self._mode == 'collect':
      # Accumulate statistics only in the collect mode, i.e. when collecting
      # data using the agent.
      for observation in observations[:, -1]:  # (batch_size, time, ...)
        # Update statistics for each observation separately for simplicity.
        # Currently during data collection the batch size is 1 anyway.
        count = running_mean_and_variance_get_count(state)
        state = fastmath.cond(
            count < self._sample_limit,
            true_operand=(observation, state),
            true_fun=lambda args: running_mean_and_variance_update(*args),
            false_operand=None,
            false_fun=lambda _: state,
        )

    mean = running_mean_and_variance_get_mean(state)
    var = running_mean_and_variance_get_variance(state)
    norm_observations = (observations - mean) / (var ** 0.5 + self._epsilon)
    self.state = state
    return norm_observations


@gin.configurable(blacklist=['mode'])
def LayerNormSquash(mode, width=128):  # pylint: disable=invalid-name
  """Dense-LayerNorm-Tanh normalizer inspired by ACME."""
  # https://github.com/deepmind/acme/blob/master/acme/jax/networks/continuous.py#L34
  del mode
  return tl.Serial([
      tl.Dense(width),
      tl.LayerNorm(),
      tl.Tanh(),
  ])
