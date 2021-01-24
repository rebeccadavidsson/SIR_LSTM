
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
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True

DEVICE = 'cpu'


class LSTM():

    def __init__(self, COUNTRY, TRAIN_UP_TO, ThreshConf, ThreshDead, target,
                 show_Figure=True):
        self.COUNTRY      = COUNTRY
        self.TRAIN_UP_TO  = TRAIN_UP_TO
        self.ThreshConf   = ThreshConf
        self.ThreshDead   = ThreshDead
        self.target       = target
        self.show_Figure  = show_Figure
        self.winSize      = 10
        self.obsSize      = 5
        self.futureSteps  = 15
        self.supPredSteps = self.winSize - self.obsSize
        self.uPredSteps   = self.futureSteps - self.supPredSteps
        self.allPredSteps = self.futureSteps + self.obsSize
        self.df           = self.init_data()

        print(f"Init LSTM model for {COUNTRY}, trained up to {TRAIN_UP_TO}, \
                with a Confirmed Cases threshold of {self.ThreshConf}  \
                and window size of {self.winSize}")

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
        
    def simulate(self):
        """
        Run a LSTM model with given parameters and return the MAPE.
        """
        print(self.ThreshConf, "threshold")
        errorData  = cc.get_nearest_sequence(self.df, self.COUNTRY,
                                        alignThreshConf=self.ThreshConf,
                                        alignThreshDead=self.ThreshDead,
                                        errorFunc  = l1_norm_error
                                        )

        confData = dataUtils.get_target_data(self.df, errorData,
                                            errorThresh = .5,
                                            country     = self.COUNTRY,
                                            target      = 'confirmed')
        deadData = dataUtils.get_target_data(self.df, errorData,
                                            errorThresh = .5, 
                                            country     = self.COUNTRY, 
                                            target      = 'fatalities')

        confScaler = dataUtils.get_scaler(confData, 'confirmed')
        deadScaler = dataUtils.get_scaler(deadData, 'fatalities')

        w = WeightInitializer()

        # build the model
        self.confModel = BasicRecurrentPredictor(
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

        w.init_weights(self.confModel, 'normal_', {})

        self.confTrainData = dataUtils.get_train_data(confData, 'confirmed', 
                                        step       = 1,
                                        winSize    = self.winSize, 
                                        trainLimit = self.TRAIN_UP_TO, 
                                        scaler     = confScaler,
                                        shuffle    = True)

        self.confLoss  = nn.SmoothL1Loss()
        gradsTrain  = self.confTrainData[:, 1:] - self.confTrainData[:, :-1] 
        confGradMax = gradsTrain.max()

        self.confGLoss   = GradientSmoothLoss(confGradMax, self.uPredSteps)
        self.confOptim = optim.LBFGS(self.confModel.parameters(), 
                                lr             = 0.05, 
                                max_iter       = 75, 
                                tolerance_grad = 1e-7, 
                                history_size   = 75
                            )
        self.confModel.to(DEVICE);
        self.confTrainData = self.confTrainData.to(DEVICE);

        pBar = tqdm(range(10))
        for i in pBar:
            loss = self.confOptim.step(self.conf_closure)
            
            # update tqdm to show loss and lr
            pBar.set_postfix({'Loss ' : loss.item(), 
                            'Lr'    : self.confOptim.param_groups[0]['lr']})
            
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
        pred   = self.confModel(confValData, future = 40).cpu().detach().numpy()
        pred   = confScaler.inverse_transform(pred[0])

        error  = l1_norm_error(pred[:confValLabel.shape[0]], confValLabel.numpy())        

        # prediction
        predDate = pd.date_range(start = self.TRAIN_UP_TO, periods=pred.shape[0])              
        # plot train data
        showTrainData = confData[confData['Province_State'] == self.COUNTRY]
        showTrainData = showTrainData[showTrainData['Date'] < self.TRAIN_UP_TO]
        
        # plot val data
        showValData = confData[confData['Province_State'] == self.COUNTRY]
        showValData = showValData[showValData['Date'] >= self.TRAIN_UP_TO]

        if self.show_Figure:
            self.plot(pred, predDate, showTrainData, showValData)

        MAPE = error.item()
        print("MAPE : %2.5f"% MAPE, ' (not normalized)')     
        return MAPE
    
    def figureOptions(self, show_Figure):
        self.show_Figure = show_Figure

    def optimizeTreshold(self, confRange):
        for val in confRange:
            self.ThreshConf = val
            print(self.ThreshConf)
            MAPE = self.simulate()
    
    def plot(self, pred, predDate, showTrainData, showValData):

        fig, ax = plt.subplots(1, 1, figsize = (9, 4))
        ax.tick_params(axis='x', rotation=45)
        fig.suptitle(self.COUNTRY + ' confirmed cases prediction')
        sns.lineplot(y = pred, x = predDate, ax = ax, linewidth=4.5)
        sns.lineplot(y = 'ConfirmedCases', x = 'Date', data = showTrainData, ax = ax, linewidth=4.5)
        sns.lineplot(y = 'ConfirmedCases', x ='Date', data = showValData, ax = ax, linewidth=4.5);
        ax.legend(['Pred', 'Train', 'Validation'])
        ax.axvline(x=self.TRAIN_UP_TO, ymin = 0.0, ymax = 1.0, linestyle='--', lw = 1, color = '#808080')
        ax.grid(True)
        plt.show()