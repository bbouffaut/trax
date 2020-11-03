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
"""Tests for metrics layers."""

from absl.testing import absltest
import numpy as np

from trax import shapes
import trax.layers as tl
from trax.layers import metrics


class MetricsTest(absltest.TestCase):

  def test_cross_entropy(self):
      """
      Calculate cross entropy.

      Args:
          self: (todo): write your description
      """
    layer = metrics._CrossEntropy()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, (9, 4, 4))

  def test_accuracy(self):
      """
      Calculate accuracy.

      Args:
          self: (todo): write your description
      """
    layer = metrics._Accuracy()
    xs = [np.ones((9, 4, 4)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, (9, 4, 4))

  def test_weighted_mean_shape(self):
      """
      Test the weighted weight shape is_weight_shape.

      Args:
          self: (todo): write your description
      """
    layer = metrics._WeightedMean()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4, 20))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_weighted_mean_semantics(self):
      """
      Calculate the mean of the model.

      Args:
          self: (todo): write your description
      """
    layer = metrics._WeightedMean()
    sample_input = np.ones((3,))
    sample_weights = np.ones((3,))
    layer.init(shapes.signature([sample_input, sample_weights]))

    x = np.array([1., 2., 3.])
    weights = np.array([1., 1., 1.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 2.)

    weights = np.array([0., 0., 1.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 3.)

    weights = np.array([1., 0., 0.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 1.)

  def test_weighted_sequence_mean_semantics(self):
      """
      Test if the weighted weighted weighted weighted weighted weighted weighted weighted weighted mean.

      Args:
          self: (todo): write your description
      """
    layer = metrics._WeightedSequenceMean()
    sample_input = np.ones((2, 3))
    sample_weights = np.ones((3,))
    full_signature = shapes.signature([sample_input, sample_weights])
    layer.init(full_signature)

    x = np.array([[1., 1., 1.], [1., 1., 0.]])
    weights = np.array([1., 1., 1.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 0.5)

    weights = np.array([1., 1., 0.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 1.)

  def test_binary_cross_entropy_loss(self):
      """
      Test if the cross entropy of a binary.

      Args:
          self: (todo): write your description
      """
    layer = tl.BinaryCrossEntropyLoss()
    xs = [np.ones((9, 1)),
          np.ones((9, 1)),
          np.ones((9, 1))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_cross_entropy_loss(self):
      """
      Calculate the loss.

      Args:
          self: (todo): write your description
      """
    layer = tl.CrossEntropyLoss()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_binary_classifier(self):
      """
      Test for classifier.

      Args:
          self: (todo): write your description
      """
    layer = metrics.BinaryClassifier()
    xs = [np.ones((9, 1))]
    y = layer(xs)
    self.assertEqual(y.shape, (9, 1))

  def test_multiclass_classifier(self):
      """
      Test for multiclass classifier.

      Args:
          self: (todo): write your description
      """
    layer = metrics.MulticlassClassifier()
    xs = [np.ones((9, 4, 4, 20))]
    y = layer(xs)
    self.assertEqual(y.shape, (9, 4, 4))

  def test_accuracy_binary_scalar(self):
      """
      Test if the accuracy is a scalar.

      Args:
          self: (todo): write your description
      """
    layer = tl.Accuracy(classifier=tl.BinaryClassifier())
    xs = [np.ones((9, 1)),
          np.ones((9, 1)),
          np.ones((9, 1))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_accuracy_multiclass_scalar(self):
      """
      Test for scalar_accuracy_scalar is a scalar.

      Args:
          self: (todo): write your description
      """
    layer = tl.Accuracy(classifier=tl.MulticlassClassifier())
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_accuracy_scalar(self):
      """
      Compute accuracy.

      Args:
          self: (todo): write your description
      """
    layer = tl.Accuracy()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_l2_loss(self):
      """
      Test the l2 loss.

      Args:
          self: (todo): write your description
      """
    layer = tl.L2Loss()
    sample_input = np.ones((2, 2))
    sample_target = np.ones((2, 2))
    sample_weights = np.ones((2, 2))
    full_signature = shapes.signature([sample_input,
                                       sample_target,
                                       sample_weights])
    layer.init(full_signature)

    x = np.array([[1., 1.], [1., 1.]])
    target = np.array([[1., 1.], [1., 0.]])
    weights = np.array([[1., 1.], [1., 0.]])
    loss = layer((x, target, weights))
    np.testing.assert_allclose(loss, 0.0)

    weights = np.array([[1., 0.], [0., 1.]])
    loss = layer((x, target, weights))
    np.testing.assert_allclose(loss, 0.5)

  def test_smooth_l1_loss(self):
      """
      Smooth loss.

      Args:
          self: (todo): write your description
      """
    layer = tl.SmoothL1Loss()
    sample_input = np.ones((2, 2))
    sample_target = np.ones((2, 2))
    sample_weights = np.ones((2, 2))
    full_signature = shapes.signature([sample_input,
                                       sample_target,
                                       sample_weights])
    layer.init(full_signature)

    x = np.array([[1., 1.], [1., 2.]])
    target = np.array([[1., 1.], [1., 0.]])
    l1_dist = 2

    weights = np.array([[1., 1.], [1., 0.]])
    loss = layer((x, target, weights))
    np.testing.assert_allclose(loss, 0.0)

    weights = np.array([[1., 0.], [0., 1.]])
    sum_weights = 2

    loss = layer((x, target, weights))
    np.testing.assert_allclose(loss, (l1_dist-0.5)/sum_weights)

    x = np.array([[1., 1.], [1., 1.5]])
    target = np.array([[1., 1.], [1., 1.]])
    l1_dist = 0.5
    loss = layer((x, target, weights))
    np.testing.assert_allclose(loss, 0.5*l1_dist**2/sum_weights)

  def test_names(self):
      """
      Method to get the namesifier

      Args:
          self: (todo): write your description
      """
    layer = tl.L2Loss()
    self.assertEqual('L2Loss_in3', str(layer))
    layer = tl.BinaryClassifier()
    self.assertEqual('BinaryClassifier', str(layer))
    layer = tl.MulticlassClassifier()
    self.assertEqual('MulticlassClassifier', str(layer))
    layer = tl.Accuracy()
    self.assertEqual('Accuracy_in3', str(layer))
    layer = tl.SequenceAccuracy()
    self.assertEqual('SequenceAccuracy_in3', str(layer))
    layer = tl.BinaryCrossEntropyLoss()
    self.assertEqual('BinaryCrossEntropyLoss_in3', str(layer))
    layer = tl.CrossEntropyLoss()
    self.assertEqual('CrossEntropyLoss_in3', str(layer))
    layer = tl.BinaryCrossEntropySum()
    self.assertEqual('BinaryCrossEntropySum_in3', str(layer))
    layer = tl.CrossEntropySum()
    self.assertEqual('CrossEntropySum_in3', str(layer))


if __name__ == '__main__':
  absltest.main()
