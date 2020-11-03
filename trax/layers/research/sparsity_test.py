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
"""Tests for trax.layers.research.efficient_attention."""

import jax
import numpy as np
from tensorflow import test

from trax import fastmath
from trax import shapes
import trax.layers as tl
from trax.layers.research import sparsity


class EfficientFeedForwardTest(test.TestCase):

  def test_blocksparse_ff_train(self):
      """
      Train blocksparse blocksparse.

      Args:
          self: (todo): write your description
      """
    d_model = 1024
    num_experts = 64
    d_ff = d_model * 8
    x_shape = (3, 7, d_model)
    with fastmath.use_backend(fastmath.Backend.JAX):
      layer = sparsity.BlockSparseFF(
          d_ff=d_ff, num_experts=num_experts, temperature=0.7, mode='train')
      x = np.ones(x_shape).astype(np.float32)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def test_blocksparse_ff_predict_equals_eval(self):
      """
      Predict a block - block - wise.

      Args:
          self: (todo): write your description
      """
    d_model = 1024
    num_experts = 64
    d_ff = d_model * 8
    x_shape = (1, 1, d_model)
    temperature = 0.7
    with fastmath.use_backend(fastmath.Backend.JAX):
      x = np.ones(x_shape).astype(np.float32)
      input_signature = shapes.signature(x)
      common_kwargs = dict(
          d_ff=d_ff,
          num_experts=num_experts,
          temperature=temperature,
      )
      eval_model = sparsity.BlockSparseFF(
          mode='eval', **common_kwargs)
      weights, state = eval_model.init(input_signature)
      eval_out, _ = eval_model.pure_fn(
          x, weights, state, rng=jax.random.PRNGKey(0))
      pred_model = sparsity.BlockSparseFF(
          mode='predict', **common_kwargs)
      _, _ = pred_model.init(input_signature)
      pred_out, _ = pred_model.pure_fn(
          x, weights, state, rng=jax.random.PRNGKey(0))
      self.assertEqual(eval_out.shape, x.shape)
      # eval_out and pred_out should be identical.
      np.testing.assert_array_almost_equal(eval_out[0, 0, :], pred_out[0, 0, :])


class ReversibleReshapePermuteTest(test.TestCase):

  def test_reversible_permute(self):
      """
      Performats the permute permute.

      Args:
          self: (todo): write your description
      """
    layer = sparsity.ReversibleReshapePermute()
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                  [0, 1, 2, 3, 4, 5, 6, 7]])
    layer.init(shapes.signature(x))
    ys = layer(x)
    self.assertEqual(tl.to_list(ys), [
        [1, 3, 5, 7, 2, 4, 6, 8],
        [0, 2, 4, 6, 1, 3, 5, 7]])
    rev_x = layer.reverse(ys, weights=layer.weights)
    self.assertEqual(tl.to_list(x), tl.to_list(rev_x))


class ReversibleRandomPermuteTest(test.TestCase):

  def test_reversible_permute(self):
      """
      Test permutation permute permute permutation.

      Args:
          self: (todo): write your description
      """
    layer = sparsity.ReversibleRandomPermute()
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13],
                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  ])
    layer.init(shapes.signature(x))
    ys = layer(x)
    # this assert will fail once per ~87B runs, but it's okay
    self.assertNotEqual(tl.to_list(ys), tl.to_list(x))

    self.assertEqual(tl.to_list(ys[0]), tl.to_list(ys[2]))
    self.assertNotEqual(tl.to_list(ys[0]), tl.to_list(ys[1]))
    rev_x = layer.reverse(ys, weights=layer.weights)
    self.assertEqual(tl.to_list(x), tl.to_list(rev_x))


class LocallyConnectedDenseTest(test.TestCase):

  def test_simple_call(self):
      """
      Call the simple simple test.

      Args:
          self: (todo): write your description
      """
    layer = sparsity.LocallyConnectedDense(2, 8)
    x = np.array([[2, 5, 3, 4],
                  [0, 1, 2, 3]])
    _, _ = layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (2, 16))


class ModularCausalAttentionTest(test.TestCase):

  def test_simple_call(self):
      """
      Call self - attention.

      Args:
          self: (todo): write your description
      """
    layer = sparsity.ModularCausalAttention(
        d_feature=4, n_heads=2, sparsity=2)
    x = np.array([[[2, 5, 3, 4],
                   [0, 1, 2, 3],
                   [0, 1, 2, 3],]])
    _, _ = layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (1, 3, 4))


class LowRankCausalAttentionTest(test.TestCase):

  def test_simple_call(self):
      """
      Perform a single layer.

      Args:
          self: (todo): write your description
      """
    layer = sparsity.LowRankCausalAttention(
        d_feature=4, n_heads=2, lowrank=2)
    x = np.array([[[2, 5, 3, 4],
                   [0, 1, 2, 3],
                   [0, 1, 2, 3],]])
    _, _ = layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (1, 3, 4))


class MultiplicativeCausalAttentionTest(test.TestCase):

  def test_simple_call(self):
      """
      Perform self attention.

      Args:
          self: (todo): write your description
      """
    layer = sparsity.MultiplicativeCausalAttention(
        d_feature=4, n_heads=2, sparsity=2)
    x = np.array([[[2, 5, 3, 4],
                   [0, 1, 2, 3],
                   [0, 1, 2, 3],]])
    _, _ = layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (1, 3, 4))


class MultiplicativeModularCausalAttentionTest(test.TestCase):

  def test_simple_call(self):
      """
      Perform layer - layer for a single layer.

      Args:
          self: (todo): write your description
      """
    layer = sparsity.MultiplicativeModularCausalAttention(
        d_feature=4, n_heads=2, sparsity=2)
    x = np.array([[[2, 5, 3, 4],
                   [0, 1, 2, 3],
                   [0, 1, 2, 3],]])
    _, _ = layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (1, 3, 4))


class CausalFavorTest(test.TestCase):

  def test_call_and_grad(self):
      """
      Perform the gradient of the gradient of the model.

      Args:
          self: (todo): write your description
      """
    layer = tl.Serial(
        tl.Dense(4),
        sparsity.CausalFavor(d_feature=4, n_heads=2),
        tl.L2Loss()
    )
    x = np.random.uniform(size=(1, 2, 4)).astype(np.float32)
    w = np.ones_like(x)
    x_sig = shapes.signature(x)
    w_sig = shapes.signature(w)
    layer.init((x_sig, x_sig, w_sig))
    y = layer((x, x, w))
    self.assertEqual(y.shape, ())
    state = layer.state
    rng = fastmath.random.get_prng(0)
    fwd = lambda weights, inp: layer.pure_fn(inp, weights, state, rng=rng)[0]
    g = fastmath.grad(fwd)(layer.weights, (x, x, w))
    self.assertEqual(g[0][0].shape, (4, 4))


if __name__ == '__main__':
  test.main()
