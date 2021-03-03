
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
from core.nn.loss   import GradientSmoothLoss

from core.networks  import BasicRecurrentPredictor

# Reproduceble results
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cpu'


class LSTM():

    def __init__(self, COUNTRY, TRAIN_UP_TO, FUTURE_DAYS, 
                ThreshDead, target, TYPE, DELAY_START, show_Figure=True):
        self.COUNTRY      = COUNTRY
        self.TRAIN_UP_TO  = TRAIN_UP_TO
        self.ThreshDead   = ThreshDead
        self.target       = target
        self.show_Figure  = show_Figure
        self.FUTURE_DAYS  = FUTURE_DAYS
        self.TYPE         = TYPE
        self.winSize      = 7
        self.obsSize      = 5
        self.futureSteps  = 15
        self.iterations   = 6
        self.supPredSteps = self.winSize - self.obsSize
        self.uPredSteps   = self.futureSteps - self.supPredSteps
        self.allPredSteps = self.futureSteps + self.obsSize
        self.bestValData  = None
        self.bestTrainData= None
        self.bestPred     = None
        self.lowestError  = 10e10
        self.SIRdicts     = []
        self.DELAY_START  = DELAY_START
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
        Run a LSTM model with given parameters and return the MAPE.
        """
        errorData = cc.get_nearest_sequence(self.df, self.COUNTRY,
                                        alignThreshConf=ThreshConf,
                                        alignThreshDead=self.ThreshDead,
                                        errorFunc      =rmsle_error
                                        )
    
        confData = dataUtils.get_target_data(self.df, errorData,
                                            errorThresh = .5,
                                            country     = self.COUNTRY,
                                            target      = 'confirmed')
        deadData = dataUtils.get_target_data(self.df, errorData,
                                            errorThresh = .5, 
                                            country     = self.COUNTRY, 
                                            target      = 'fatalities')

        if input_data is not None:
            temp_df = confData
            predictions = input_data["pred"]
            date = input_data["TRAIN_UP_TO"] - datetime.timedelta(days=1) # - datetime.timedelta(days=input_data["FUTURE_DAYS"])
            if second:
                date = date - datetime.timedelta(days=second)
                print(date, len(predictions))
            # display(confData[confData["Date"] == self.TRAIN_UP_TO])
            for pred in predictions:
                temp_df.loc[temp_df["Date"] == date, "ConfirmedCases"] = pred
                date += datetime.timedelta(days=1)
            confData = temp_df
            # self.TRAIN_UP_TO = date - datetime.timedelta(days = int(input_data["FUTURE_DAYS"] / 1))
            self.TRAIN_UP_TO = date
        print(f"Init LSTM model for {self.COUNTRY}, trained up to {self.TRAIN_UP_TO}, with a Confirmed Cases threshold of {round(ThreshConf)}  and window size of {self.winSize}")


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

        status = "ok"
        pBar = tqdm(range(self.iterations))
        for i in pBar:
            loss = self.confOptim.step(self.conf_closure)
            if loss > 10:
                print(loss)
                status = "fail"
                self.simulate(input_data=input_data, second=second)
                break
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
        self.pred   = self.confModel(confValData, future = self.FUTURE_DAYS).cpu().detach().numpy()
        self.pred   = confScaler.inverse_transform(self.pred[0])

        error  = rmsle_error(self.pred[:confValLabel.shape[0]], confValLabel.numpy())        

        # prediction
        self.predDate = pd.date_range(start = self.TRAIN_UP_TO, periods=self.pred.shape[0])              
        # plot train data
        self.showTrainData = confData[confData['Province_State'] == self.COUNTRY]
        self.showTrainData = self.showTrainData[self.showTrainData['Date'] < self.TRAIN_UP_TO]
        
        # plot val data
        self.showValData = confData[confData['Province_State'] == self.COUNTRY]
        self.showValData = self.showValData[self.showValData['Date'] >= self.TRAIN_UP_TO]

        error = error.item()
        if math.isnan(error):
            status = "fail"
            error = 10e10

        if error <= self.lowestError or self.bestValData is None:
            self.bestValData   = self.showValData
            self.bestTrainData = self.showTrainData
            self.bestPred      = self.pred
            self.lowestError   = error

        if self.show_Figure:
            self.plot()

        if input_data is not None:
            pred = list(input_data["pred"]) + list(self.bestPred)
        else:
            pred = self.bestPred
        print("RMSLE : %2.5f"% error, ' (not normalized)')     
        return {"error": error, 'valData': self.bestValData, 
                'trainData': self.bestTrainData, 'pred': pred, 
                "TRAIN_UP_TO": self.TRAIN_UP_TO, "FUTURE_DAYS": self.FUTURE_DAYS}

    def rollingUpdate(self):
        """
        For the rolling update mechanism, if update is true,
        predictions (self.pred) are added to confData at the right location
        self.TRAIN_UP_TO then also gets shifted.
        """
        ITERATIONS = 2
        # First run
        for i in range(ITERATIONS):
            results = self.simulate()

        self.clear_history()
        for i in range(ITERATIONS, input=results):
            results = self.simulate()
        return results



    def clear_history(self):
        self.bestValData  = None
        self.bestTrainData= None
        self.bestPred     = None

    def figureOptions(self, show_Figure):
        self.show_Figure = show_Figure

    def add_df(self, df, name):
        self.SIRdicts.append(df)

    def _optimizeTreshold(self, confRange):
        self.iterations = 5
        results_dict = {}
        for threshold in confRange:
            print(threshold)
            # Calculate mean of 5 runs
            error_list = []
            for i in range(5):
                self.ThreshConf = threshold
                error = self.simulate(threshold, show_Figure=False)
                error_list.append(error)
            results_dict[threshold] = np.mean(error_list)
            print(results_dict)
        return results_dict
    
    def optimizeTreshold(self):

        best = fmin(self.simulate,
            space=hp.uniform('Threshold', 50, 105),
            algo=tpe.suggest,
            max_evals=5)
        return best

    
    def plot(self):
        showValData, showTrainData = self.bestValData, self.bestTrainData
        pred, predDate = self.bestPred, self.predDate
        date_until = self.TRAIN_UP_TO + datetime.timedelta(days=self.FUTURE_DAYS)
        showValData = showValData[showValData["Date"] < date_until + datetime.timedelta(days=20)]
        fig, ax = plt.subplots(1, 1, figsize = (9, 4))
        ax.tick_params(axis='x', rotation=45)
        fig.suptitle(self.COUNTRY + ' confirmed cases prediction')
        sns.lineplot(y = 'ConfirmedCases', x ='Date', data = showValData, ax = ax, linewidth=4.5);
        sns.lineplot(y = pred, x = predDate, ax = ax, linewidth=4.5)
        sns.lineplot(y = 'ConfirmedCases', x = 'Date', data = showTrainData, ax = ax, linewidth=4.5)
        
        if len(self.SIRdicts) > 0:
            SIRdata, SIRDdata, SIRFdata = pd.DataFrame(self.SIRdicts[0]), pd.DataFrame(self.SIRdicts[1]), pd.DataFrame(self.SIRdicts[2])
            SIRdata = SIRdata[SIRdata["Date"] < date_until]
            SIRFdata = SIRFdata[SIRFdata["Date"] < date_until]
            SIRDdata = SIRDdata[SIRDdata["Date"] < date_until]
            sns.lineplot(y = 'New Confirmed', x ='Date', data = SIRdata, ax = ax, linewidth=4.5);
            sns.lineplot(y = 'New Confirmed', x ='Date', data = SIRDdata, ax = ax, linewidth=4.5);
            sns.lineplot(y = 'New Confirmed', x ='Date', data = SIRFdata, ax = ax, linewidth=4.5);
            ax.legend(['Validation', 'LSTM', 'Train', 'SIR', 'SIRD', 'SIRF'])
        else:
            ax.legend(['Validation', 'Pred', 'Train'])
        ax.axvline(x=self.TRAIN_UP_TO, ymin = 0.0, ymax = 1.0, linestyle='--', lw = 1, color = '#808080')
        ax.grid(True)
        plt.show()

    def accuracy_types(self):

        end_date = self.TRAIN_UP_TO + datetime.timedelta(days=self.FUTURE_DAYS)
        showValData, showTrainData = self.bestValData, self.bestTrainData
        showValData = showValData[showValData["Date"] < end_date]
    
        dfOverTime = pd.DataFrame()
        total_rmse, total_mse = [], []
        for i in range(1, len(showValData) - 1):
            y, ypred = showValData["ConfirmedCases"].iloc[0:i],  self.bestPred[0:i]
            mse = mean_squared_error(y, ypred, squared=False)
            total_mse.append(mse)
            rmse = mean_squared_error(y, ypred, squared=True)
            total_rmse.append(rmse)

            dfOverTime = pd.DataFrame()
            dfOverTime["mse"] = total_mse
            dfOverTime["rmse"] = total_rmse
            
        return dfOverTime

    
    def accuracy(self):
        end_date = self.TRAIN_UP_TO + datetime.timedelta(days=self.FUTURE_DAYS)
        showValData, showTrainData = self.bestValData, self.bestTrainData
        showValData = showValData[showValData["Date"] < end_date]

        SIRdata, SIRDdata, SIRFdata = pd.DataFrame(self.SIRdicts[0]), pd.DataFrame(self.SIRdicts[1]), pd.DataFrame(self.SIRdicts[2])
        SIRdata = SIRdata[SIRdata["Date"] >= self.TRAIN_UP_TO]
        SIRdata = SIRdata[SIRdata["Date"] <= end_date - datetime.timedelta(days=1)]
        SIRDdata = SIRDdata[SIRDdata["Date"] >= self.TRAIN_UP_TO]
        SIRDdata = SIRDdata[SIRDdata["Date"] <= end_date - datetime.timedelta(days=1)]
        SIRFdata = SIRFdata[SIRFdata["Date"] >= self.TRAIN_UP_TO]
        SIRFdata = SIRFdata[SIRFdata["Date"] <= end_date - datetime.timedelta(days=1)]
        SIRdata = SIRdata.rename(columns={"New Confirmed": "New Confirmed SIR"})
        SIRDdata = SIRDdata.rename(columns={"New Confirmed": "New Confirmed SIRD"})
        SIRFdata = SIRFdata.rename(columns={"New Confirmed": "New Confirmed SIRF"})
    
        squared = False
        if SIRdata is None:
            dfOverTime = pd.DataFrame(columns=["lstm"])
            y, ypred = showValData["ConfirmedCases"], self.bestPred
            mse = mean_squared_error(y, ypred, squared=squared)
            pd.DataFrame([[mse]], columns=["lstm"])
        else:
            total_mse, total_mse_SIR, total_mse_SIRD, total_mse_SIRF = [], [], [], []

            predictions = pd.DataFrame()
            predictions["Date"] = SIRdata["Date"]
            predictions["Pred"] = self.bestPred

            merged = pd.merge(SIRdata, showValData, how="inner", on="Date")
            merged = pd.merge(SIRDdata, merged, how="inner", on="Date")
            merged = pd.merge(SIRFdata, merged, how="inner", on="Date")
            merged = pd.merge(merged, predictions, on="Date", how="inner")
            
            y, ypred = merged["ConfirmedCases"], merged["Pred"]
            mse = mean_squared_error(y, ypred, squared=squared)
            y, ypred = merged["ConfirmedCases"], merged["New Confirmed SIR"]
            mse_SIR = mean_squared_error(y, ypred, squared=squared)
            y, ypred = merged["ConfirmedCases"], merged["New Confirmed SIRD"]
            mse_SIRD = mean_squared_error(y, ypred, squared=squared)
            y, ypred = merged["ConfirmedCases"], merged["New Confirmed SIRF"]
            mse_SIRF = mean_squared_error(y, ypred, squared=squared)
            df = pd.DataFrame([[mse, mse_SIR, mse_SIRD, mse_SIRF]], columns=["lstm", "SIR", "SIRD", "SIRF"])

            for i in range(1, len(showValData) - 1):
                y, ypred = merged["ConfirmedCases"].iloc[0:i], merged["Pred"][0:i]
                mse = mean_squared_error(y, ypred, squared=squared)
                total_mse.append(mse)
                y, ypred = merged["ConfirmedCases"].iloc[0:i], merged["New Confirmed SIR"].iloc[0:i]
                mse_SIR = mean_squared_error(y, ypred, squared=squared)
                total_mse_SIR.append(mse_SIR)
                y, ypred = merged["ConfirmedCases"].iloc[0:i], merged["New Confirmed SIRD"].iloc[0:i]
                mse_SIRD = mean_squared_error(y, ypred, squared=squared)
                total_mse_SIRD.append(mse_SIRD)
                y, ypred = merged["ConfirmedCases"].iloc[0:i], merged["New Confirmed SIRF"].iloc[0:i]
                mse_SIRF = mean_squared_error(y, ypred, squared=squared)
                total_mse_SIRF.append(mse_SIRF)
            dfOverTime = pd.DataFrame()
            dfOverTime["lstm"] = total_mse
            dfOverTime["SIR"] = total_mse_SIR
            dfOverTime["SIRD"] = total_mse_SIRD
            dfOverTime["SIRF"] = total_mse_SIRF
            
        return df, dfOverTime
        