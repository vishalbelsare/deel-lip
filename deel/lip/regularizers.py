# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import warnings
from tensorflow.keras.regularizers  import Regularizer
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from .utils import _deel_export

def compute_orth_dist(w, stride = 2, padding = 1,verbose=False):  #use_stochastic= True,
    [R,R1,i_c, o_c] = w.shape
    if verbose:
        print("stride ",stride)
        print("padding ",padding)
        print("w.shape ",w.shape)
    w_reshape = tf.transpose(w,perm=[3,0,1,2])
    if verbose:
        print("w_reshape ",w_reshape.shape)
    w_padded = tf.pad(w_reshape,paddings=[[0,0],[padding,padding],[padding,padding],[0,0]])
    if verbose:
        print("w_padded ",w_padded.shape)
    output = tf.nn.conv2d(w_padded, w, strides=[1,stride,stride,1], padding='VALID')
    if verbose:
        print(output.shape)
    outm3 = output.shape[-3]
    outm2 = output.shape[-2]
    target = tf.zeros((o_c, outm3, outm2, o_c))
    if verbose:
        print("target shape ",target.shape, " with Id (o_c,o_c) ",o_c)
    ct = int(np.floor(outm2/2))
    
    target_zeros = tf.zeros((outm3*outm2-1,o_c,  o_c))
    if verbose:
        print(target_zeros[:ct*outm2+ct].shape)
        print(tf.expand_dims(tf.eye(o_c), axis=0).shape)
        print(target_zeros[ct*outm2+ct:].shape)
    target = tf.concat([target_zeros[:ct*outm2+ct],tf.expand_dims(tf.eye(o_c), axis=0),target_zeros[ct*outm2+ct:]],axis=0)
    
    if verbose:
        print("target ",target.shape)
    target = tf.reshape(target,(outm3,outm2,o_c,o_c))
    
    if verbose:
        print("target ",target.shape)
    target = tf.transpose(target,[2,0,1,3])
    if verbose:
        print("target ",target.shape)
        print("output ",output.shape)
        tf.print(output[:,ct,ct,:])
    return tf.reduce_sum(tf.square( output - target ))


@_deel_export
class LorthRegularizer(Regularizer):
    def __init__(
        self,
        kernel_shape=None,
        stride=1,
        lambdaLorth=1.0,
        dim=2, ## 2 for 2D conv, 1 for 1D conv
        flag_deconv=False,
    ) -> None:
        """
        Regularize a conv kernel to be orthogonal (sigma min and max =1) using Lorth regularizer

        Args:

        """
        super(LorthRegularizer, self).__init__()
        self.stride = stride
        self.lambdaLorth = lambdaLorth
        self.kernel_shape = kernel_shape
        self.dim = dim
        self.flag_deconv = flag_deconv
        self.r = 0 ## will be set by set_kernel_shape
        self.delta = 0 ## will be set by set_kernel_shape
        self.set_kernel_shape(self.kernel_shape)
    def get_kernel_shape(self):
        if self.dim == 2:
            (R,R,C,M) = self.kernel_shape
        else:
            raise NotImplementedError
            (R,C,M) = self.kernel_shape
        return (R,C,M)
    def compute_delta(self):
        delta = 0.
        (R,C,M)=self.get_kernel_shape()
        if self.flag_deconv==False: 
            delta = M-(self.stride**self.dim)*C
        else:
            delta = C-(self.stride**self.dim)*M
        delta = max(0,delta)
        if delta > 0:
            print("Embedding case C=",C," M=",M," Stride=",self.stride," deconv=",self.flag_deconv," => Minimum delta: ", delta)
        return delta
    def check_if_orthconv_exists(self):
        assert True, "check_if_orthconv_exists Not implemented"
        (R,C,M) = self.get_kernel_shape()
        ##RO case 
        if C*self.stride**self.dim>=M:
            if M > C*(R**self.dim):
                raise RuntimeError("Impossible RO configuration for orthogonal convolution")
        else:
            if self.stride > R:
                raise RuntimeError("Impossible CO configuration for orthogonal convolution")

        if C*(self.stride**self.dim)==M:
            warnings.warn("LorthRegularizer: Warning configuration C*S^2=M is hard to optimize")
            
    def set_kernel_shape(self, shape):
        if shape is None:
            return
        self.kernel_shape = shape
        (R,C,M) = self.get_kernel_shape()
        assert R&1, "Lorth Regularizer require odd kernels "+str(R)
        self.r = R//2
        self.padding = ((R-1)//self.stride)*self.stride ## padding size for Lorth convolution
        self.delta = self.compute_delta()
        self.check_if_orthconv_exists() 

    def __call__(self, x):
        reg = self.lambdaLorth*(compute_orth_dist(x,padding = self.padding, stride=self.stride)-self.delta)
        return reg
        
    def get_config(self):
        return {
            "kernel_shape": self.kernel_shape,
            "stride": self.stride,
            "lambdaLorth": self.lambdaLorth,
            "dim": self.dim,
            "flag_deconv": self.flag_deconv,
        }


@_deel_export
class OrthDenseRegularizer(Regularizer):
    def __init__(
        self,
        lambdaOrth=1.0,
    ) -> None:
        """
        Regularize a Dense kernel to be orthogonal (sigma min and max =1) minimizing W.W^T-Id

        Args:

        """
        super(OrthDenseRegularizer, self).__init__()
        self.lambdaOrth = lambdaOrth
    
    def denseOrthDist(self,w):
        transp_b = w.shape[0]<=w.shape[1]
        #print(w.shape)
        wwt = tf.matmul(w, w,  transpose_a = not transp_b,  transpose_b = transp_b)  #WW^T if h<=w; W^TW otherwise
        #print(wwt.shape)
        idx = tf.eye(wwt.shape[0])
        return tf.reduce_sum(tf.square( wwt - idx ))

    def __call__(self, x):
        reg = self.lambdaOrth*self.denseOrthDist(x)
        return reg
        
    def get_config(self):
        return {
            "lambdaOrth": self.lambdaOrth,
        }

