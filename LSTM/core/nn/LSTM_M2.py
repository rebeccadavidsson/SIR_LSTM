
import numpy as np
import covsirphy as cs
from core.networks import BasicRecurrentPredictor
from core.nn.loss import GradientSmoothLoss
from core.nn.loss import mape_error
from core.nn.loss import r2
from core.nn.loss import rmsle_
from core.nn.loss import mae_error
from core.nn.loss import rmsle_error
from core.nn.loss import l1_norm_error
from core.nn import WeightInitializer
from core.data import utils as dataUtils
from core.data import compare_countries as cc
from hyperopt import fmin, tpe, hp, STATUS_OK
import math
import datetime
from scipy import optimize
from torch.optim import lr_scheduler
from torch import optim
from torch import nn
from IPython.display import display
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import statistics
import torch
import time
import warnings
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

warnings.filterwarnings('ignore')


# Reproduceble results
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cpu'
TRAIN_UP_TO = pd.to_datetime('2020-10-01')

PARAMS_DICT = {
    "window":  7,
    "steps": 5,
}

TARGET = "New Confirmed"
TYPE = "LSTMCell"
THRESHDEAD = 20


class LSTM_M2():

    def __init__(self, COUNTRY, TRAIN_UP_TO=TRAIN_UP_TO, FUTURE_DAYS=50,
                 DELAY_START=TRAIN_UP_TO, hiddenCells=16, dropoutRate=0,
                 iterations=5, PARAMS=PARAMS_DICT, N=10e7,
                 show_Figure=False):
        if COUNTRY == "United States":
            self.COUNTRY = "US"
        else:
            self.COUNTRY = COUNTRY
        self.TRAIN_UP_TO = TRAIN_UP_TO
        self.ThreshDead = THRESHDEAD
        self.target = TARGET
        self.show_Figure = show_Figure
        self.FUTURE_DAYS = FUTURE_DAYS
        self.TYPE = TYPE
        self.hiddenCells = hiddenCells
        self.winSize = PARAMS["window"]
        self.obsSize = PARAMS["steps"]
        self.futureSteps = 15
        self.dropoutRate = dropoutRate
        self.iterations = iterations
        self.supPredSteps = self.winSize - self.obsSize
        self.uPredSteps = self.futureSteps - self.supPredSteps
        self.allPredSteps = self.futureSteps + self.obsSize
        self.bestValData = None
        self.bestTrainData = None
        self.bestPred = None
        self.lowestError = 10e10
        self.SIRdicts = []
        self.DELAY_START = DELAY_START
        self.N = N
        self.df = self.init_data()

    def init_data(self):
        df = pd.read_csv('assets/covid_spread.csv', parse_dates=['Date'])
        df = dataUtils.preprocess_data(df)

        # Replace Confirmed cases with new confirmed cases
        if self.target == "New Confirmed":
            df = df.sort_values("Date")
            df["New Confirmed"] = df.groupby("Province_State")[
                "ConfirmedCases"].diff()
            df = df.groupby('Province_State', as_index=False).apply(
                lambda group: group.iloc[1:-1])
            df = df.reset_index()
            df = df.drop("ConfirmedCases", axis=1)
            df = df.rename(columns={"New Confirmed": "ConfirmedCases"})
        return df

    def input_data(self, new_df):
        df = self.df
        df = df[df["Province_State"] == self.COUNTRY]
        new_df = new_df[new_df["Date"] <= pd.to_datetime(
            df["Date"].values[-1]).strftime("%Y-%m-%d")]
        new_df = new_df[new_df["Date"] >= pd.to_datetime(
            df["Date"].values[0]).strftime("%Y-%m-%d")]
        if len(new_df) != len(df):
            raise ValueError('length of dfs are not compatible')

        df["ConfirmedCases"] = new_df["Infected"].values
        df["Fatalities"] = new_df["Fatal"].values

        # Replace old values
        self.df = self.df.drop(
            self.df[self.df["Country_Region"] == self.COUNTRY].index)
        self.df = self.df.drop(
            self.df[self.df["Province_State"] == self.COUNTRY].index)
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
                                            errorFunc=rmsle_error
                                            )

        errorThresh = 0.5
        confData = dataUtils.get_target_data(self.df, errorData,
                                             errorThresh=errorThresh,
                                             country=self.COUNTRY,
                                             target='confirmed')

        confScaler = dataUtils.get_scaler(confData, 'confirmed')

        w = WeightInitializer()

        # build the model
        self.confModel = BasicRecurrentPredictor(
            # parameters
            chNo=1,          # number of input features
            future=0,
            returnFullSeq=True,     # return both the encoded sequence
            # and the future prediction

            # RNN
            rnnCell=self.TYPE,  # RNN cell type (LSTM/GRU/RNN)
            rnnNoCells=1,          # no of RNN cells
            hidChNo=self.hiddenCells,         # number of RNN cell hidden dimension

            # MLP
            mlpLayerCfg=[4],      # layer hidden dims
            mlpActiv='PReLU',  # inner activation of the mlp
            dropRate=self.dropoutRate,     # dropout rate for each layer of mlp
            normType=None,     # normalization type
            mlpActivLast=None      # note that every timestamp
            # in the sequence will be activated too
        ).build()

        w.init_weights(self.confModel, 'normal_', {})

        self.confTrainData = dataUtils.get_train_data(confData, 'confirmed',
                                                      step=5,
                                                      winSize=self.winSize,
                                                      trainLimit=self.TRAIN_UP_TO,
                                                      scaler=confScaler,
                                                      shuffle=True)

        self.confLoss = nn.SmoothL1Loss()

        gradsTrain = self.confTrainData[:, 1:] - self.confTrainData[:, :-1]
        confGradMax = gradsTrain.max()

        self.confGLoss = GradientSmoothLoss(confGradMax, self.uPredSteps)
        self.confOptim = optim.LBFGS(self.confModel.parameters(),
                                     lr=0.1,
                                     max_iter=75,
                                     tolerance_grad=1e-7,
                                     history_size=75
                                     )
        self.confModel.to(DEVICE)
        self.confTrainData = self.confTrainData.to(DEVICE)

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
        self.pred = self.confModel(
            confValData, future=self.FUTURE_DAYS).cpu().detach().numpy()
        self.pred = confScaler.inverse_transform(self.pred[0])

        error = rmsle_error(
            self.pred[:confValLabel.shape[0]], confValLabel.numpy())
        errors = [rmsle_(self.pred[:confValLabel.shape[0]], confValLabel.numpy(), self.N),
                  mae_error(self.pred[:confValLabel.shape[0]],
                            confValLabel.numpy(), self.N),
                  mape_error(self.pred[:confValLabel.shape[0]], confValLabel.numpy())]

        # prediction
        self.predDate = pd.date_range(
            start=self.TRAIN_UP_TO, periods=self.pred.shape[0])
        # plot train data
        self.showTrainData = confData[confData['Province_State']
                                      == self.COUNTRY]
        self.showTrainData = self.showTrainData[self.showTrainData['Date']
                                                < self.TRAIN_UP_TO]

        # plot val data
        self.showValData = confData[confData['Province_State'] == self.COUNTRY]
        self.showValData = self.showValData[self.showValData['Date']
                                            >= self.TRAIN_UP_TO]

        error = error.item()

        if math.isnan(error):
            error = 10e10
            self.simulate(input_data=input_data)
        elif error > 1.5:
            self.simulate(input_data=input_data)

        if error <= self.lowestError or self.bestValData is None:
            self.bestValData = self.showValData
            self.bestTrainData = self.showTrainData
            self.bestPred = self.pred
            self.lowestError = error

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

    def plot(self):
        showValData, showTrainData = self.bestValData, self.bestTrainData
        pred, predDate = self.bestPred, self.predDate
        date_until = self.TRAIN_UP_TO + \
            datetime.timedelta(days=self.FUTURE_DAYS)
        showValData = showValData[showValData["Date"]
                                  < date_until + datetime.timedelta(days=20)]
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        ax.tick_params(axis='x', rotation=45)
        fig.suptitle(self.COUNTRY + ' confirmed cases prediction')
        sns.lineplot(y='ConfirmedCases', x='Date',
                     data=showValData, ax=ax, linewidth=4.5)
        sns.lineplot(y=pred, x=predDate, ax=ax, linewidth=4.5)
        sns.lineplot(y='ConfirmedCases', x='Date',
                     data=showTrainData, ax=ax, linewidth=4.5)

        if len(self.SIRdicts) > 0:
            SIRdata, SIRDdata, SIRFdata = pd.DataFrame(self.SIRdicts[0]), pd.DataFrame(
                self.SIRdicts[1]), pd.DataFrame(self.SIRdicts[2])
            SIRdata = SIRdata[SIRdata["Date"] < date_until]
            SIRFdata = SIRFdata[SIRFdata["Date"] < date_until]
            SIRDdata = SIRDdata[SIRDdata["Date"] < date_until]
            sns.lineplot(y='New Confirmed', x='Date',
                         data=SIRdata, ax=ax, linewidth=4.5)
            sns.lineplot(y='New Confirmed', x='Date',
                         data=SIRDdata, ax=ax, linewidth=4.5)
            sns.lineplot(y='New Confirmed', x='Date',
                         data=SIRFdata, ax=ax, linewidth=4.5)
            ax.legend(['Validation', 'LSTM', 'Train', 'SIR', 'SIRD', 'SIRF'])
        else:
            ax.legend(['Validation', 'Pred', 'Train'])
        ax.axvline(x=self.TRAIN_UP_TO, ymin=0.0, ymax=1.0,
                   linestyle='--', lw=1, color='#808080')
        ax.grid(True)
        plt.show()

    def get_periods(self, nums):
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))

    def estimate_country(self, jhu_data, population_data, oxcgrt_data, NPI):
        COUNTRY = self.COUNTRY
        NPI_df = oxcgrt_data.cleaned()
        NPI_df = NPI_df[NPI_df["Country"] == COUNTRY]

        # Get lockdown dates
        NPI_df = NPI_df.reset_index().drop('index', axis=1)
        NPI_df = NPI_df.groupby("Date").mean().reset_index()

        s = cs.Scenario(jhu_data, population_data, country=COUNTRY)
        days_delay, df_periods = s.estimate_delay(oxcgrt_data)

        # NPI_dates = {}
        # min_rate = 75

        # if NPI != "Stringency_index":
        #     min_rate = 3
        # periods = []
        # while periods == [] or min_rate >= 1:
        #     lockdown_indexes = NPI_df[NPI_df[NPI] >= min_rate].index
        #     lockdown_dates = NPI_df[NPI_df[NPI] >= min_rate]["Date"]
        #     periods = self.get_periods(lockdown_indexes)
        #     min_rate -= 1
        # lockdown_dates_adjusted = []
        # for date in lockdown_dates:
        #     new_date = date + datetime.timedelta(days = days_delay)
        #     lockdown_dates_adjusted.append(new_date)
        # NPI_dates[NPI] = lockdown_dates_adjusted
        # lockdown_dates_adjusted = pd.Series(lockdown_dates_adjusted)
        # start = periods[-1][1]

        # if start > 410:
        #     start = periods[-1][0]
        # print(start, periods, len(NPI_df))
        # if periods == [] or start < 90:
        #     return False, False, False, False

        # DELAY_START = NPI_df.iloc[start + days_delay].Date

        # index = -1
        # while DELAY_START >= pd.to_datetime("2021-03-01"):
        #     DELAY_START = NPI_df.iloc[periods[-index][0]].Date
        #     index -= 1

        NPI_dates = {}
        lockdown_indexes = NPI_df[NPI_df[NPI] >= 75].index
        lockdown_dates = NPI_df[NPI_df[NPI] >= 75]["Date"]
        periods = self.get_periods(lockdown_indexes)

        if periods == []:
            lockdown_indexes = NPI_df[NPI_df[NPI] >= 65].index
            lockdown_dates = NPI_df[NPI_df[NPI] >= 65]["Date"]
            periods = self.get_periods(lockdown_indexes)

        if periods == []:
            lockdown_indexes = NPI_df[NPI_df[NPI] >= 4].index
            lockdown_dates = NPI_df[NPI_df[NPI] >= 4]["Date"]
            periods = self.get_periods(lockdown_indexes)

        if periods == []:
            lockdown_indexes = NPI_df[NPI_df[NPI] >= 3].index
            lockdown_dates = NPI_df[NPI_df[NPI] >= 3]["Date"]
            periods = self.get_periods(lockdown_indexes)

        lockdown_dates_adjusted = []
        for date in lockdown_dates:
            new_date = date + datetime.timedelta(days=days_delay)
            lockdown_dates_adjusted.append(new_date)
        NPI_dates[NPI] = lockdown_dates_adjusted

        lockdown_dates_adjusted = []
        for date in lockdown_dates:
            new_date = date + datetime.timedelta(days=days_delay)
            lockdown_dates_adjusted.append(new_date)
        lockdown_dates_adjusted = pd.Series(lockdown_dates_adjusted)

        # Save variable DELAY_START, which is equal to the start of a lockdown period
        # in this case, we will hardcode the script to the second lockdown date
        # if len(periods) > 2:
        print(len(periods), periods, "PERIODS")
        if periods == []:
            return False, False, False, False
        DELAY_START = NPI_df.iloc[periods[len(periods) - 1][0]].Date
        print(DELAY_START)

        if COUNTRY == "Italy":
            DELAY_START = pd.to_datetime('2020-12-21')

        if COUNTRY == "United Kingdom":
            df_params = pd.read_pickle("./data/df_United_Kingdom")
        elif COUNTRY == "Sweden":
            df_params = pd.read_pickle("../figures/pickles/df_Sweden")
        elif COUNTRY == "United States":
            df_params = pd.read_pickle("../figures/pickles/df_United_States")
        else:
            df_params = pd.read_pickle("../figures/pickles/df_9_countries")
        df_params = df_params[df_params["Country"] == COUNTRY]

        return DELAY_START, df_params, NPI_dates, days_delay

    def compute_errors(self, N, df):
        errors = pd.DataFrame(columns=["RMSE", "MAE", "MAPE"])

        observed = df["Observed"]
        pred = df["M2"]
        errors_list = [rmsle_(pred, observed, N),
                       mae_error(pred, observed, N),
                       mape_error(pred, observed)]
        errors = errors.append(
            pd.Series(errors_list, index=errors.columns), ignore_index=True)
        return errors
