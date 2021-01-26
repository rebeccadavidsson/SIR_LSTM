from core.nn.LSTMmodel import LSTM
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle
import datetime
import seaborn as sns
import covsirphy as cs
from core.SIR import SIR
sns.set()


DEVICE       = 'cpu'
# TRAIN_UP_TO  = pd.to_datetime('2020-10-01')
ThreshConf   = 68
ThreshDead   = 20
target       = "New Confirmed"
FUTURE_DAYS  = 40


class CompareModels():

    def __init__(self, TRAIN_UP_TO):
        self.TRAIN_UP_TO = pd.to_datetime(TRAIN_UP_TO)

    def runCompare(self, COUNTRY):
        """
        Return accuracy over time for an LSTM and SIRF model
        """
        lstm = LSTM(COUNTRY, self.TRAIN_UP_TO, FUTURE_DAYS, ThreshDead, target)
        lstm.init_data()

        # Optimize threshold
        lstm.iterations = 5
        best = lstm.optimizeTreshold()
        lstm.iterations = 10
        ThreshConf = best["Threshold"]

        lstm.simulate(ThreshConf=ThreshConf)
        sir_sim = SIR(COUNTRY, self.TRAIN_UP_TO)
        snl = sir_sim.init_data()
        summary = snl.summary()
        summary["Start_dt"] = pd.to_datetime(summary["Start"], format="%d%b%Y")
        summary["End_dt"] = pd.to_datetime(summary["End"], format="%d%b%Y")
        query = summary[summary["End_dt"] > self.TRAIN_UP_TO]
        all_phases = query.index.tolist()

        snl.combine(phases=all_phases)
        target_date = datetime.datetime.strftime(self.TRAIN_UP_TO - datetime.timedelta(days=1), format="%d%b%Y")
        snl.separate(target_date)
        summary = snl.summary()
        summary = snl.summary()
        all_phases = summary.index.tolist()
        snl.disable(phases=all_phases[:-1])
        snl.enable(phases=all_phases[-1:])

        snl.estimate(model=cs.SIRF)
        df = snl.simulate();
        df["New Confirmed"] = df["Confirmed"].diff()

        lstm.plot(SIRdata=df)
        accuracy, dfOverTime = lstm.accuracy(SIRdata=df)
        return dfOverTime

