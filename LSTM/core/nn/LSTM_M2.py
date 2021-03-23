
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
from sklearn.metrics       import mean_squared_error
from sklearn.preprocessing import StandardScaler

from tqdm             import tqdm
from IPython.display  import display

from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from scipy import optimize
import datetime
import math
from hyperopt import fmin, tpe, hp, STATUS_OK

from core.data      import compare_countries as cc
from core.data      import utils             as dataUtils

from core.nn        import WeightInitializer
from core.nn.loss   import l1_norm_error
from core.nn.loss   import rmsle_error
from core.nn.loss   import mae_error
from core.nn.loss   import rmsle_
from core.nn.loss   import r2
from core.nn.loss   import mape_error
from core.nn.loss   import GradientSmoothLoss

from core.networks  import BasicRecurrentPredictor

# Reproduceble results
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cpu'


PARAMS_DICT = {
    "window":  7,
    "steps": 5,
}

class LSTM_M2():

    def __init__(self, COUNTRY, TRAIN_UP_TO, FUTURE_DAYS, 
                ThreshDead=20, target="New Confirmed", 
                TYPE="LSTMCell", DELAY_START=14, PARAMS=PARAMS_DICT, N=10e7,
                show_Figure=False):
        if COUNTRY == "United States":
            self.COUNTRY = "US"
        else:
            self.COUNTRY = COUNTRY
        self.TRAIN_UP_TO  = TRAIN_UP_TO
        self.ThreshDead   = 0
        self.target       = target
        self.show_Figure  = show_Figure
        self.FUTURE_DAYS  = FUTURE_DAYS
        self.TYPE         = TYPE
        self.winSize      = PARAMS["window"]
        self.obsSize      = PARAMS["steps"]
        self.futureSteps  = 15
        self.iterations   = 5
        self.supPredSteps = self.winSize - self.obsSize
        self.uPredSteps   = self.futureSteps - self.supPredSteps
        self.allPredSteps = self.futureSteps + self.obsSize
        self.bestValData  = None
        self.bestTrainData= None
        self.bestPred     = None
        self.lowestError  = 10e10
        self.SIRdicts     = []
        self.DELAY_START  = DELAY_START
        self.N            = N
        self.df           = self.init_data()

    def init_data(self):
        df = pd.read_csv('assets/covid_spread.csv', parse_dates=['Date'])
        df = dataUtils.preprocess_data(df)

        # Replace Confirmed cases with new confirmed cases
        if self.target == "New Confirmed":
            df = df.sort_values("Date")
            df["New Confirmed"] = df.groupby("Province_State")["ConfirmedCases"].diff()
            df = df.groupby('Province_State', as_index=False).apply(lambda group: group.iloc[1:-1])
            df = df.reset_index()
            df = df.drop("ConfirmedCases", axis=1)
            df = df.rename(columns={"New Confirmed": "ConfirmedCases"})
        return df
    
    def input_data(self, new_df):
        df = self.df
        df = df[df["Country_Region"] == self.COUNTRY]
        new_df = new_df[new_df["Date"] <= pd.to_datetime(df["Date"].values[-1]).strftime("%Y-%m-%d")]
        new_df = new_df[new_df["Date"] >= pd.to_datetime(df["Date"].values[0]).strftime("%Y-%m-%d")]
        if len(new_df) != len(df):
            raise ValueError('length of dfs are not compatible')
    
        df["ConfirmedCases"] = new_df["Infected"].values
        df["Fatalities"] = new_df["Fatal"].values
        
        # Replace old values
        self.df = self.df.drop(self.df[self.df["Country_Region"] == self.COUNTRY].index)
        self.df = self.df.drop(self.df[self.df["Province_State"] == self.COUNTRY].index)
        self.df = self.df.append(df)
        return


    def conf_closure(self):
        self.confOptim.zero_grad()
        self.confModel.returnFullSeq = True
        
        # slice data
        obsData = self.confTrainData[:, :self.obsSize]
        
        # make prediction
        out = self.confModel(obsData, future=self.futureSteps)
        out = out.reshape(-1, self.allPredSteps, 1)
        
        # compute gradients
        loss = self.confLoss(out[:, :self.winSize], self.confTrainData)
        
        # unsupervised loss
        smoothLoss = self.confGLoss(out[:, self.winSize:], 0.25)
        loss += smoothLoss
        
        # make prediciton follow an ascending trend
        # by forcing the gradients to be positie (still testing)
        grads = out[:, 1:] - out[:, :-1]
        grads[grads > 0] = 0
        grads = grads.mean().abs()
        loss += grads
        loss.backward()
        
        # clip gradients / numerical stability
        nn.utils.clip_grad_norm_(self.confModel.parameters(), 1.0)
        
        return loss

    def simulate(self, input_data=None, second=False, ThreshConf=70, pred=None):
        """
        Run a LSTM model with given parameters and return the specified error.
        """

        errorData = cc.get_nearest_sequence(self.df, self.COUNTRY,
                                        alignThreshConf=0,
                                        alignThreshDead=self.ThreshDead,
                                        errorFunc      =rmsle_error
                                        )

        errorThresh = 1
        confData = dataUtils.get_target_data(self.df, errorData,
                                            errorThresh = errorThresh,
                                            country     = self.COUNTRY,
                                            target      = 'confirmed')
            
        confScaler = dataUtils.get_scaler(confData, 'confirmed')


        w = WeightInitializer()

        # build the model
        self.confModel = BasicRecurrentPredictor(
                    # parameters
                    chNo        = 1,          # number of input features
                    future      = 0,
                    returnFullSeq = True,     # return both the encoded sequence 
                                            # and the future prediction
            
                    # RNN
                    rnnCell     = self.TYPE, # RNN cell type (LSTM/GRU/RNN)
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

        w.init_weights(self.confModel, 'normal_', {})   

        self.confTrainData = dataUtils.get_train_data(confData, 'confirmed', 
                                        step       = 5,
                                        winSize    = self.winSize, 
                                        trainLimit = self.TRAIN_UP_TO, 
                                        scaler     = confScaler,
                                        shuffle    = True)

        self.confLoss  = nn.SmoothL1Loss()

        gradsTrain  = self.confTrainData[:, 1:] - self.confTrainData[:, :-1] 
        confGradMax = gradsTrain.max()

        self.confGLoss   = GradientSmoothLoss(confGradMax, self.uPredSteps)
        self.confOptim = optim.LBFGS(self.confModel.parameters(), 
                                lr             = 0.1, 
                                max_iter       = 75, 
                                tolerance_grad = 1e-7, 
                                history_size   = 75
                            )
        self.confModel.to(DEVICE)
        self.confTrainData = self.confTrainData.to(DEVICE);


        for i in range(self.iterations):
            loss = self.confOptim.step(self.conf_closure)
            if loss > 10:
                break

            if torch.isnan(loss):
                raise ValueError('Loss is NaN')
        
        confValData, confValLabel = dataUtils.get_val_data(confData, 'confirmed', 
                                                        self.COUNTRY, 
                                                        self.TRAIN_UP_TO, 
                                                        self.obsSize, 
                                                        confScaler)
        confValData = confValData.to(DEVICE)

        self.confModel.eval()

        # make prediction
        self.confModel.returnFullSeq = False
        self.pred   = self.confModel(confValData, future = self.FUTURE_DAYS).cpu().detach().numpy()
        self.pred   = confScaler.inverse_transform(self.pred[0])

        error  = rmsle_error(self.pred[:confValLabel.shape[0]], confValLabel.numpy())       
        errors =  [rmsle_(self.pred[:confValLabel.shape[0]], confValLabel.numpy(), self.N), 
                    mae_error(self.pred[:confValLabel.shape[0]], confValLabel.numpy(), self.N),
                    mape_error(self.pred[:confValLabel.shape[0]], confValLabel.numpy())]

        # prediction
        self.predDate = pd.date_range(start = self.TRAIN_UP_TO, periods=self.pred.shape[0])              
        # plot train data
        self.showTrainData = confData[confData['Province_State'] == self.COUNTRY]
        self.showTrainData = self.showTrainData[self.showTrainData['Date'] < self.TRAIN_UP_TO]
        
        # plot val data
        self.showValData = confData[confData['Province_State'] == self.COUNTRY]
        self.showValData = self.showValData[self.showValData['Date'] >= self.TRAIN_UP_TO]

        error = error.item()
        print(error)
        if math.isnan(error):
            error = 10e10
            self.simulate(input_data=input_data)
        elif error > 1:
            self.simulate(input_data=input_data)

        if error <= self.lowestError or self.bestValData is None:
            self.bestValData   = self.showValData
            self.bestTrainData = self.showTrainData
            self.bestPred      = self.pred
            self.lowestError   = error

        if self.show_Figure:
            self.plot()

        return {"error": error, "errors": errors, 'valData': self.bestValData, 'confData': confData,
                'trainData': self.bestTrainData, 'pred': self.bestPred, 
                "TRAIN_UP_TO": self.TRAIN_UP_TO, "FUTURE_DAYS": self.FUTURE_DAYS}

    def update_predictions(self, SIR_pred, tau):
        if self.bestPred is None:
            raise ValueError("Use simulate() to compute predictions")

        if len(SIR_pred) != len(self.bestPred):
            raise ValueError("Input predictions are not compatible")
        new_pred = []
        weights = np.linspace(2, 1, tau)
        i = 0
        for pred, pred_SIR in zip(self.bestPred, SIR_pred):
            weight = 1
            if i < tau and i >= 0:
                weight = weights[i] 
            new_pred.append((pred * weight + pred_SIR) / (1 + weight))
            i += 1
        return new_pred
