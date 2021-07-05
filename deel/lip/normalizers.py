# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains computation function, for Bjorck and spectral
normalization. This is done for internal use only.
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from .utils import padding_circular, transposeKernel, zero_upscale2D

DEFAULT_NITER_BJORCK = 15
DEFAULT_NITER_SPECTRAL = 3
DEFAULT_NITER_SPECTRAL_INIT = 10
DEFAULT_BETA_BJORCK = 0.25


def reshaped_kernel_orthogonalization(
    kernel,
    u,
    adjustment_coef,
    niter_spectral=DEFAULT_NITER_SPECTRAL,
    niter_bjorck=DEFAULT_NITER_BJORCK,
    beta=DEFAULT_BETA_BJORCK,
):
    """
    Perform reshaped kernel orthogonalization (RKO) to the kernel given as input. It
    apply the power method to find the largest singular value and apply the Bjorck
    algorithm to the rescaled kernel. This greatly improve the stability and and
    speed convergence of the bjorck algorithm.

    Args:
        kernel: the kernel to orthogonalize
        u: the vector used to do the power iteration method
        adjustment_coef: the adjustment coefficient as used in convolution
        niter_spectral: number of iteration to do in spectral algorithm
        niter_bjorck: iteration used for bjorck algorithm
        beta: the beta used in the bjorck algorithm

    Returns: the orthogonalized kernel, the new u, and sigma which is the largest
        singular value

    """
    W_bar, u, sigma = spectral_normalization(kernel, u, niter=niter_spectral)
    W_bar = bjorck_normalization(W_bar, niter=niter_bjorck, beta=beta)
    W_bar = W_bar * adjustment_coef
    W_bar = K.reshape(W_bar, kernel.shape)
    return W_bar, u, sigma


def bjorck_normalization(w, niter=DEFAULT_NITER_BJORCK, beta=DEFAULT_BETA_BJORCK):
    """
    apply Bjorck normalization on w.

    Args:
        w: weight to normalize, in order to work properly, we must have
            max_eigenval(w) ~= 1
        niter: number of iterations
        beta: beta used in each iteration, must be in the interval ]0, 0.5]

    Returns:
        the orthonormal weights

    """
    for i in range(niter):
        w = (1 + beta) * w - beta * w @ tf.transpose(w) @ w
    return w


def _power_iteration(w, u, niter=DEFAULT_NITER_SPECTRAL):
    """
    Internal function that performs the power iteration algorithm.

    Args:
        w: weights matrix that we want to find eigen vector
        u: initialization of the eigen vector
        niter: number of iteration, must be greater than 0

    Returns:
         u and v corresponding to the maximum eigenvalue

    """
    _u = u
    for i in range(niter):
        _v = tf.math.l2_normalize(_u @ tf.transpose(w))
        _u = tf.math.l2_normalize(_v @ w)
    return _u, _v


def spectral_normalization(kernel, u, niter=DEFAULT_NITER_SPECTRAL):
    """
    Normalize the kernel to have it's max eigenvalue == 1.

    Args:
        kernel: the kernel to normalize
        u: initialization for the max eigen vector
        niter: number of iteration

    Returns:
        the normalized kernel w_bar, it's shape, the maximum eigen vector, and the
        maximum eigen value

    """
    W_shape = kernel.shape
    if u is None:
        niter *= 2  # if u was not known increase number of iterations
        u = tf.ones(shape=tuple([1, W_shape[-1]]))
    # Flatten the Tensor
    W_reshaped = tf.reshape(kernel, [-1, W_shape[-1]])
    _u, _v = _power_iteration(W_reshaped, u, niter)
    # Calculate Sigma
    sigma = _v @ W_reshaped
    sigma = sigma @ tf.transpose(_u)
    # normalize it
    W_bar = W_reshaped / sigma
    return W_bar, _u, sigma



def _power_iteration_conv(w, u, stride = 1.0, conv_first = True, cPad=None, niter=DEFAULT_NITER_SPECTRAL, bigConstant=-1):
    """
    Internal function that performs the power iteration algorithm.

    Args:
        w: weights matrix that we want to find eigen vector
        u: initialization of the eigen vector
        niter: number of iteration, must be greater than 0

    Returns:
         u and v corresponding to the maximum eigenvalue

    """
    def iter_f(u):
        u=u/tf.norm(u)
        if cPad is None:
            padType = 'SAME'
        else:
            padType='VALID'

        if conv_first:
            u_pad=padding_circular(u,cPad)
            v= tf.nn.conv2d(u_pad,w,padding=padType,strides=(1,stride,stride,1))
            v1 = zero_upscale2D(v,(stride,stride))
            v1=padding_circular(v1,cPad)
            wAdj=transposeKernel(w,True)
            unew=tf.nn.conv2d(v1,wAdj,padding=padType,strides=1)
        else:
            u1 = zero_upscale2D(u,(stride,stride))
            u_pad=padding_circular(u1,cPad)
            wAdj=transposeKernel(w,True)
            v=tf.nn.conv2d(u_pad,wAdj,padding=padType,strides=1)
            v1=padding_circular(v,cPad)
            unew= tf.nn.conv2d(v1,w,padding=padType,strides=(1,stride,stride,1))
        if bigConstant> 0:
            unew = bigConstant*u-unew
        return unew,v

    _u = u
    for i in range(niter):
        _u,_v = iter_f(_u)
    return _u, _v

@tf.function
def spectral_normalization_conv(kernel, u=None, stride = 1.0, conv_first = True, cPad=None, niter=DEFAULT_NITER_SPECTRAL):
    """
    Normalize the convolution kernel to have it's max eigenvalue == 1.

    Args:
        kernel: the convolution kernel to normalize
        u: initialization for the max eigen matrix
        stride: stride parameter of convolutuions
        conv_first: RO or CO case stride^2*C<M
        cPad: Circular padding (k//2,k//2)
        niter: number of iteration

    Returns:
        the normalized kernel w_bar, it's shape, the maximum eigen vector, and the
        maximum eigen value

    """
    '''W_shape = kernel.shape
    if u is None:
        niter *= 2  # if u was not known increase number of iterations
        u = K.random_normal(shape=tuple([1, W_shape[-1]]))
    # Flatten the Tensor
    W_reshaped = K.reshape(kernel, [-1, W_shape[-1]])'''
    if niter <= 0:
        return kernel, u, 1.0
    _u, _v = _power_iteration_conv(kernel, u, stride = stride, conv_first = conv_first, cPad=cPad, niter=niter)
    # Calculate Sigma
    sigma = tf.norm(_v)
    W_bar = kernel / sigma
    return W_bar, _u, sigma

