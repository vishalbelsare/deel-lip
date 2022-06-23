# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
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
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

import warnings
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.utils import register_keras_serializable


class Lorth(ABC):
    def __init__(self, dim, kernel_shape=None, stride=1, conv_transpose=False) -> None:
        """
        Base class for Lorth regularization. Not meant to be used standalone.

        Args:
            dim (int): the rank of the convolution, e.g. "2" for 2D convolution.
            kernel_shape: the shape of the kernel.
            stride (int): stride used in the associated convolution
            conv_transpose (bool): whether the kernel is from a transposed convolution.
        """
        super(Lorth, self).__init__()
        self.dim = dim
        self.stride = stride
        self.conv_transpose = conv_transpose
        self.set_kernel_shape(kernel_shape)

    def _get_kernel_shape(self):
        """Return the kernel size, the number of input channels and output channels"""
        return [self.kernel_shape[i] for i in (0, -2, -1)]

    def _compute_delta(self):
        """delta is positive in CO case, zero in RO case."""
        _, C, M = self._get_kernel_shape()
        if not self.conv_transpose:
            delta = M - (self.stride**self.dim) * C
        else:
            delta = C - (self.stride**self.dim) * M
        return max(0, delta)

    def _check_if_orthconv_exists(self):
        R, C, M = self._get_kernel_shape()
        msg = "Impossible {} configuration for orthogonal convolution."
        if C * self.stride**self.dim >= M:  # RO case
            if M > C * (R**self.dim):
                raise RuntimeError(msg.format("RO"))
        else:  # CO case
            if self.stride > R:
                raise RuntimeError(msg.format("CO"))
        if C * (self.stride**self.dim) == M:  # square case
            warnings.warn(
                "LorthRegularizer: warning configuration C*S^2=M is hard to optimize."
            )

    def set_kernel_shape(self, shape):
        if shape is None:
            self.kernel_shape, self.padding, self.delta = None, None, None
            return

        R = shape[0]
        self.kernel_shape = shape
        self.padding = ((R - 1) // self.stride) * self.stride
        self.delta = self._compute_delta()

        # Assertions on kernel shape and existence of orthogonal convolution
        assert R & 1, "Lorth regularizer requires odd kernels. Receives " + str(R)
        self._check_if_orthconv_exists()

    @abstractmethod
    def _compute_conv_kk(self, w):
        raise NotImplementedError()

    @abstractmethod
    def _compute_target(self, w, output_shape):
        raise NotImplementedError()

    def compute_lorth(self, w):
        output = self._compute_conv_kk(w)
        target = self._compute_target(w, output.shape)
        return tf.reduce_sum(tf.square(output - target)) - self.delta


class Lorth2D(Lorth):
    def __init__(self, kernel_shape=None, stride=1, conv_transpose=False) -> None:
        """
        Lorth computation for 2D convolutions. Although this class allows to compute
        the regularization term, it cannot be used as it is in a layer.

        Args:
            kernel_shape: the shape of the kernel.
            stride (int): stride used in the associated convolution
            conv_transpose (bool): whether the kernel is from a transposed convolution.
        """
        dim = 2
        super(Lorth2D, self).__init__(dim, kernel_shape, stride, conv_transpose)

    def _compute_conv_kk(self, w):
        w_reshape = tf.transpose(w, perm=[3, 0, 1, 2])
        w_padded = tf.pad(
            w_reshape,
            paddings=[
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
        )
        return tf.nn.conv2d(w_padded, w, self.stride, padding="VALID")

    def _compute_target(self, w, convKxK_shape):
        C_out = w.shape[-1]
        outm3 = convKxK_shape[-3]
        outm2 = convKxK_shape[-2]
        ct = tf.cast(tf.math.floor(outm2 / 2), dtype=tf.int32)

        target_zeros = tf.zeros((outm3 * outm2 - 1, C_out, C_out))
        target = tf.concat(
            [
                target_zeros[: ct * outm2 + ct],
                tf.expand_dims(tf.eye(C_out), axis=0),
                target_zeros[ct * outm2 + ct :],
            ],
            axis=0,
        )

        target = tf.reshape(target, (outm3, outm2, C_out, C_out))
        target = tf.transpose(target, [2, 0, 1, 3])
        return target
