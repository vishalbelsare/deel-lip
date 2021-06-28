
import numpy as np
import tensorflow as tf
from .utils import _deel_export, padding_circular
from .layers import LipschitzLayer, Condensable
import deel.lip

from tensorflow.keras.layers import (
    Layer,
    Dense,
    Conv2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)

## Not the best place and no the best code => may replace by wandb
def printAndLog(txt, log_out=None):
    print(txt)
    if log_out is not None:
        print(txt,file=log_out)

def zero_upscale2D(x,strides):
    stride_v = strides[0]*strides[1]
    if stride_v==1:
        return x
    output_shape = x.get_shape().as_list()[1:]
    if strides[1]>1:
        output_shape[1]*=strides[1]
        x = tf.expand_dims(x, 3)
        fillz = tf.zeros_like(x)
        fillz = tf.tile(fillz,[1,1,1,strides[1]-1,1])
        x = tf.concat((x, fillz), axis=3)
        x= tf.reshape(x,(-1,)+tuple(output_shape))
    if strides[0]>1:
        output_shape[0]*=strides[0]
        x = tf.expand_dims(x, 2)
        fillz = tf.zeros_like(x)
        fillz = tf.tile(fillz,[1,1,strides[0]-1,1,1])
        x = tf.concat((x, fillz), axis=2)
        x= tf.reshape(x,(-1,)+tuple(output_shape))
    return x

def transposeKernel(w, transpose=False):
    if not transpose:
        return w
    wAdj=tf.transpose(w,perm=[0,1,3,2])
    wAdj=wAdj[::-1,::-1,:]
    return wAdj

def compute_layer_vs_2D(w,Ks,N,nbIter):    
    def model_iter_conf(w,input_shape,conv_first = True, cPad=None):
        def f(u, bigConstant=-1):
            u=u/tf.norm(u)
            if cPad is None:
                padType = 'SAME'
            else:
                padType='VALID'

            if conv_first:
                u_pad=padding_circular(u,cPad)
                v= tf.nn.conv2d(u_pad,w,padding=padType,strides=(1,Ks,Ks,1))
                v1 = zero_upscale2D(v,(Ks,Ks))
                v1=padding_circular(v1,cPad)
                wAdj=transposeKernel(w,True)
                unew=tf.nn.conv2d(v1,wAdj,padding=padType,strides=1)
            else:
                u1 = zero_upscale2D(u,(Ks,Ks))
                u_pad=padding_circular(u1,cPad)
                wAdj=transposeKernel(w,True)
                v=tf.nn.conv2d(u_pad,wAdj,padding=padType,strides=1)
                v1=padding_circular(v,cPad)
                unew= tf.nn.conv2d(v1,w,padding=padType,strides=(1,Ks,Ks,1))
            if bigConstant> 0:
                unew = bigConstant*u-unew
            return unew,v
        return f
    (R0,R,d,D) = w.shape
    KN = int(Ks*N)
    batch_size = 1
    cPad=[int(R0/2),int(R/2)]

    
    
    if Ks*Ks*d > D:
        input_shape=(N,N,D)
        iter_f=model_iter_conf(w,(batch_size,)+input_shape,conv_first = False,cPad=cPad)
        premier = 1
        second = 0
    else:
        input_shape=(KN,KN,d)
        iter_f=model_iter_conf(w,(batch_size,)+input_shape,conv_first = True,cPad=cPad)
        premier = 0
        second = 1

    # Maximum singular value 
    
    
    u=tf.random.uniform((batch_size,)+input_shape,minval=-1.0, maxval=1.0)

    for it in range(nbIter):
        u,v = iter_f(u) 

    sigma_max = tf.norm(v) #norm_u(v)
    
    # Minimum Singular Value
    
    bigConstant = 1.1 * sigma_max**2
    print("bigConstant "+str(bigConstant))
    u=tf.random.uniform((batch_size,)+input_shape,minval=-1.0, maxval=1.0)

    for it in range(nbIter): 
        u, v = iter_f(u, bigConstant=bigConstant)   
        
    if bigConstant - tf.norm(u) >= 0:                      # cas normal
        sigma_min = tf.sqrt( bigConstant - tf.norm(u) )
    elif bigConstant - tf.norm(u) >= -0.0000000000001:    # précaution pour gérer les erreurs numériques
        sigma_min=0
    else:
        sigma_min = -1                                    # veut dire qu'il y a un prolème
    
    return (float(sigma_min) , float(sigma_max))

def computeDenseSV(layer,input_size, numIter=100, log_out=None):
    weights=np.copy(layer.get_weights()[0])
    #print(weights.shape)
    kernel_n=weights.astype(dtype='float32')
    printAndLog('----------------------------------------------------------',log_out)
    printAndLog('Layer type '+str(type(layer))+" weight shape "+str(weights.shape),log_out)
    new_w = np.reshape(weights, [weights.shape[0],-1])
    svdtmp = np.linalg.svd(new_w, compute_uv=False)
    SVmin = np.min(svdtmp)
    SVmax = np.max(svdtmp)
    printAndLog('kernel(W) SV (min, max, mean) '+str((SVmin,SVmax,np.mean(svdtmp))),log_out)
    return (SVmin,SVmax)

def computeConvSV(layer,input_size, numIter=100, log_out=None):
    Ks = layer.strides[0]
    #isLinear = False
    weights=np.copy(layer.get_weights()[0])
    #print(weights.shape)
    kernel_n=weights.astype(dtype='float32')
    printAndLog('----------------------------------------------------------',log_out)
    printAndLog('Layer type '+str(type(layer))+" weight shape "+str(weights.shape),log_out)
    if Ks>1:
        #print("Warning np.linalg.svd incompatible with strides")
        SVmin,SVmax = compute_layer_vs_2D(weights,Ks,input_size,numIter)
        printAndLog("Conv(K) SV min et max [conv iter]: "+str((SVmin,SVmax)),log_out)
         #out_stats['conv SV (min, max, mean)']=(vs[0],vs[1],-1234)
    else:
        kernel_n=weights.astype(dtype='float32')
        transforms = np.fft.fft2(kernel_n, (input_size,input_size), axes=[0, 1])
        svd = np.linalg.svd(transforms, compute_uv=False)    
        SVmin = np.min(svd)
        SVmax = np.max(svd)
        printAndLog("Conv(K) SV min et max [np.linalg.svd]: "+str((SVmin,SVmax)),log_out)
        printAndLog("Conv(K) SV mean et std [np.linalg.svd]: "+str((np.mean(svd),np.std(svd))),log_out)
    #print("SV ",np.sort(np.reshape(svd,(-1,))))
    return (SVmin,SVmax)

### Warning this is not SV for non linear functions but grad min and grad max
def computeActivationSV(layer,input_size, numIter=100, log_out=None):
    if isinstance(layer,tf.keras.layers.ReLU):
        return (0,1)
    if isinstance(layer,deel.lip.activations.GroupSort):
        return (1,1)
    if isinstance(layer,deel.lip.activations.MaxMin):
        return (1,1)
    
def computeLayerSV(layer,input_size=-1, numIter=100, log_out=None, supplementaryType2SV={}):
    defaultType2SV = {
        tf.keras.layers.Conv2D: computeConvSV,
        tf.keras.layers.Conv2DTranspose: computeConvSV,
        deel.lip.layers.SpectralConv2D: computeConvSV,
        deel.lip.layers.FrobeniusConv2D: computeConvSV,
        deel.lip.layers.LorthRegulConv2D: computeConvSV,
        tf.keras.layers.Dense: computeDenseSV,
        deel.lip.layers.SpectralDense: computeDenseSV,
        deel.lip.layers.FrobeniusDense: computeDenseSV,
    }
    src_layer = layer
    if isinstance(layer, Condensable):
        printAndLog("vanilla_export",log_out)
        printAndLog(str(type(layer)),log_out)
        layer.condense()
        layer = layer.vanilla_export()
    if type(layer) in defaultType2SV.keys():
        return defaultType2SV[type(layer)](layer,src_layer.input_shape[1],numIter=numIter,log_out=log_out)
    elif type(layer) in supplementaryType2SV.keys():
        return supplementaryType2SV[type(layer)](layer,src_layer.input_shape[1],numIter=numIter,log_out=log_out)
    else:
        printAndLog("No SV for layer "+str(type(layer)),log_out)
        return (None,None)
  
def computeModelSVs(model,input_size=-1, numIter=100, log_out=None, supplementaryType2SV={}):
    list_SV = []
    for layer in model.layers:
        if isinstance(layer,tf.keras.models.Model) or isinstance(layer,tf.keras.models.Sequential):
            list_SV.append((layer.name , (None,None)))
            list_SV += computeModelSVs(layer,input_size=input_size, numIter=numIter, log_out=log_out, supplementaryType2SV=supplementaryType2SV)
        else:
            list_SV.append((layer.name , computeLayerSV(layer,input_size=input_size, numIter=numIter, log_out=log_out, supplementaryType2SV=supplementaryType2SV)))
    return list_SV

def computeModelUpperLip(model,input_size=-1, numIter=100, log_out=None, supplementaryType2SV={}):
    list_SV = computeModelSVs(model,input_size=-1, numIter=100, log_out=None, supplementaryType2SV={})
    UpperLip = 1.0
    LowerLip = 1.0
    count_nb_notknown = 0
    for svs in list_SV:
        if svs[1][0] is not None:
            UpperLip*=svs[1][1]
            LowerLip*=svs[1][0]
        else:
            count_nb_notknown+=1
    printAndLog("Total lower and upper gradient bound: "+str(LowerLip)+", "+str(UpperLip),log_out)
    return ['Total',(LowerLip,UpperLip)]  +  list_SV
 


