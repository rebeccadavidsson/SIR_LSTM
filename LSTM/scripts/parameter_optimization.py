
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import warnings 
warnings.filterwarnings('ignore')
import time
import torch
import statistics

import numpy   as np 
import pandas  as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics       import mean_squared_log_error
from sklearn.preprocessing import StandardScaler

from tqdm             import tqdm
from IPython.display  import display 

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from core.data      import compare_countries as cc
from core.data      import utils             as dataUtils

from core.nn        import WeightInitializer
from core.nn.loss   import l1_norm_error
from core.nn.loss   import GradientSmoothLoss

from core.networks  import BasicRecurrentPredictor

# Reproduceble results
torch.manual_seed(123);
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True

COUNTRY      = 'Germany'
DEVICE       = 'cpu'
TRAIN_UP_TO  = pd.to_datetime('2020-10-01')
ThreshConf   = 500
ThreshDead   = 20
target       = "New Confirmed"
TRAIN_UP_TO  = pd.to_datetime('2020-10-01')

allData = pd.read_csv('../assets/covid_spread.csv', parse_dates=['Date'])
# Add data to Province_State
allData = dataUtils.preprocess_data(allData)

if target == "New Confirmed":
    # Replace Confirmed cases with new confirmed cases
    allData = allData.sort_values("Date")
    allData["New Confirmed"] = allData.groupby("Country_Region")["ConfirmedCases"].diff()
    columns = allData.columns
    allData = allData.groupby('Country_Region', as_index=False).apply(lambda group: group.iloc[1:])
    allData = allData.reset_index()[columns]
    allData = allData.drop("ConfirmedCases", axis=1)
    allData = allData.rename(columns={"New Confirmed": "ConfirmedCases"})

def main(alignThreshConf, alignThreshDead):
    errorData  = cc.get_nearest_sequence(allData, COUNTRY,
                                        alignThreshConf = alignThreshConf,
                                        alignThreshDead = alignThreshDead, 
                                        errorFunc       = l1_norm_error
                                        )

    confData = dataUtils.get_target_data(allData, errorData, 
                                     errorThresh = .5, 
                                     country     = COUNTRY, 
                                     target      = 'confirmed')
    deadData = dataUtils.get_target_data(allData, errorData, 
                                        errorThresh = .5, 
                                        country     = COUNTRY, 
                                        target      = 'fatalities')

    confScaler = dataUtils.get_scaler(confData, 'confirmed')
    deadScaler = dataUtils.get_scaler(deadData, 'fatalities')

    w = WeightInitializer()
    # build the model
    confModel = BasicRecurrentPredictor(
                # parameters
                chNo        = 1,          # number of input features
                future      = 0,
                returnFullSeq = True,     # return both the encoded sequence 
                                        # and the future prediction
        
                # RNN
                rnnCell     = 'LSTMCell', # RNN cell type (LSTM/GRU/RNN)
                rnnNoCells  = 1,          # no of RNN cells
                hidChNo     = 16,         # number of RNN cell hidden dimension
                
                # MLP
                mlpLayerCfg   = [4],      # layer hidden dims
                mlpActiv      = 'PReLU',  # inner activation of the mlp
                dropRate      = None,     # dropout rate for each layer of mlp
                normType      = None,     # normalization type
                mlpActivLast  = None      # note that every timestamp 
                                        # in the sequence will be activated too
                
                ).build()
    w.init_weights(confModel, 'normal_', {})
    winSize       = 10
    obsSize       = 5
    futureSteps   = 15
    supPredSteps  = winSize - obsSize
    uPredSteps    = futureSteps - supPredSteps
    allPredSteps  = futureSteps + obsSize

    confTrainData = dataUtils.get_train_data(confData, 'confirmed', 
                                    step       = 1,
                                    winSize    = winSize, 
                                    trainLimit = TRAIN_UP_TO, 
                                    scaler     = confScaler,
                                    shuffle    = True)
    confLoss  = nn.SmoothL1Loss()
    gradsTrain  = confTrainData[:, 1:] - confTrainData[:, :-1] 
    confGradMax = gradsTrain.max()

    confGLoss   = GradientSmoothLoss(confGradMax, uPredSteps)
    confOptim = optim.LBFGS(confModel.parameters(), 
                            lr             = 0.05, 
                            max_iter       = 75, 
                            tolerance_grad = 1e-7, 
                            history_size   = 75
                        )
    confModel.to(DEVICE);
    confTrainData = confTrainData.to(DEVICE);

    def conf_closure():
        confOptim.zero_grad()
        confModel.returnFullSeq = True
        
        # slice data
        obsData = confTrainData[:,:obsSize]
        
        # make prediction
        out  = confModel(obsData, future = futureSteps)
        out  = out.reshape(-1, allPredSteps, 1)
        
        # compute gradients
        loss = confLoss(out[:, :winSize], confTrainData)
        
        # unsupervised loss
        smoothLoss = confGLoss(out[:,winSize:], 0.25)
        loss += smoothLoss 
        
        # make prediciton follow an ascending trend
        # by forcing the gradients to be positie (still testing)
        grads = out[:, 1:] - out[:, :-1]
        grads[grads > 0] = 0
        grads = grads.mean().abs()
        loss += grads
        loss.backward()
        
        # clip gradients / numerical stability
        nn.utils.clip_grad_norm_(confModel.parameters(), 1.0)
        
        return loss

    pBar = tqdm(range(10))
    for i in pBar:
        loss = confOptim.step(conf_closure)
        
        # update tqdm to show loss and lr
        pBar.set_postfix({'Loss ' : loss.item(), 
                        'Lr'    : confOptim.param_groups[0]['lr']})
        
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')
    confValData, confValLabel = dataUtils.get_val_data(confData, 'confirmed', 
                                                   COUNTRY, 
                                                   TRAIN_UP_TO, 
                                                   obsSize, 
                                                   confScaler)
    confValData = confValData.to(DEVICE)

    confModel.eval()
    # get figure
    fig, ax = plt.subplots(1, 1, figsize = (9, 4))
    ax.tick_params(axis='x', rotation=45)
    fig.suptitle(COUNTRY + ' confirmed cases prediction')

    # make prediction
    confModel.returnFullSeq = False
    pred   = confModel(confValData, future = 30).cpu().detach().numpy()
    pred   = confScaler.inverse_transform(pred[0])

    error  = l1_norm_error(pred[:confValLabel.shape[0]], confValLabel.numpy())
    # print("MAPE : %2.5f"% error.item(), ' (not normalized)')             

    # prediction
    predDate = pd.date_range(start = TRAIN_UP_TO, periods=pred.shape[0])              
    sns.lineplot(y = pred, x = predDate, ax = ax, linewidth=4.5)

    # plot train data
    showTrainData = confData[confData['Province_State'] == COUNTRY]
    showTrainData = showTrainData[showTrainData['Date'] < TRAIN_UP_TO]
    sns.lineplot(y = 'ConfirmedCases', x = 'Date', data = showTrainData, ax = ax, linewidth=4.5)

    # plot val data
    showValData = confData[confData['Province_State'] == COUNTRY]
    showValData = showValData[showValData['Date'] >= TRAIN_UP_TO]
    sns.lineplot(y = 'ConfirmedCases', x ='Date', data = showValData, ax = ax, linewidth=4.5);

    ax.legend(['Pred', 'Train', 'Validation'])
    ax.axvline(x=TRAIN_UP_TO, ymin = 0.0, ymax = 1.0, linestyle='--', lw = 1, color = '#808080')
    ax.grid(True)
    return error.item(), ThreshConf


tresholdsConf = [200, 500]
tresholdsDead = [20, 20]
for ThreshConf, ThreshDead in zip(tresholdsConf, tresholdsDead):
    print(main(ThreshConf, ThreshDead))