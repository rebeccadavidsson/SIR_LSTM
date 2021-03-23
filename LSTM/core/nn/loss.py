import numpy as np

import torch
from torch import nn
from sklearn.metrics import r2_score
import scipy
import matplotlib.pylab as plt

def mape_error(y_pred, y_true):
    # y_pred=y_pred[:len(y_true)]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 

# =============================================== L1 NORM =====================================================
def l1_norm_error(source, candidate):
    # candidate = candidate[:len(source)]
    error = np.abs(source - candidate)
    source[source == 0] = 1e-30  
    error = error / source       
    error = error.mean()
    return error

def mae_error(source, candidate, N):
    # candidate = candidate[:len(source)]
    # candidate =  candidate / N
    # source = source / N
    error = np.abs(source - candidate)
    source[source == 0] = 1e-30
    error = error / source
    error = error.mean()
    return (error / N) * 1000000


# =============================================== RMSLE  =====================================================
def rmsle_error(source, candidate):
    # candidate = candidate[:len(source)]
    candidate += 1e-30
    error = np.log10((source + 1) / (candidate + 1))
    error = error * error
    error = error.mean()
    error = np.sqrt(error)
    return error

def rmsle_(source, candidate, N=1):
    # candidate = candidate[:len(source)]
    candidate += 1e-30
    error = (source) - (candidate)
    error = error * error
    error = error.mean()
    error = np.sqrt(error)
    return (error / N)

# =============================================== R2  =====================================================

def r2(pred, y_true):
    return r2_score(pred, y_true[:len(pred)])



# =============================================== GRADIENT SMOOTH =====================================================
class GradientSmoothLoss(nn.Module):
    def __init__(self, refGrad, future, decayFunc=None):
        '''
        Function that minimizes the rate of change of a time series prediction,
        as the times evolves. It tries to give a desired "shape".

        :param refGrad:   the maximum gradient that is used for scaling
        :param future:    number of future predictions in the timeseries
        :param decayFunc: decay function for weights (the weights decrease as time increases, such that the last
                            timestamps will have a smoother rate of change)
        '''

        super().__init__()
        self.future  = future
        self.refGrad = refGrad

        # compute decay weights
        decay = np.linspace(0, 1, future)
        decay = self.__linear_decay(decay) if decayFunc is None \
                                            else decayFunc(decay)
        decay = torch.from_numpy(decay)

        self.decay    = decay * refGrad

    # =============================================== LINEAR DECAY =====================================================
    def __linear_decay(self, linSpace):
        return 0.8 - linSpace * 0.5

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor, clampVal = 0.25):
        '''
        :param inTensor: input tensor on which to apply the loss
        :param clampVal: clamp errors before averaging for better stability
        :return:
        '''

        self.decay = self.decay.to(inTensor.device)

        gradOut = inTensor[:, 1:] - inTensor[:, :-1]
        gradOut = gradOut.abs() - self.decay
        gradOut = torch.clamp(gradOut, min=0, max=clampVal)
        gradOut = gradOut.mean()

        return gradOut


