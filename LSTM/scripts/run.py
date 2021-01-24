
from LSTMmodel import LSTM
import pandas as pd

COUNTRY      = 'Netherlands'
DEVICE       = 'cpu'
TRAIN_UP_TO  = pd.to_datetime('2020-10-01')
ThreshConf   = 100
ThreshDead   = 20
target       = "New Confirmed"
confRange    = [10, 500, 700]

lstm = LSTM(COUNTRY, TRAIN_UP_TO, ThreshConf, ThreshDead, target)
lstm.simulate()

# lstm.figureOptions(show_Figure=False)
# lstm.optimizeTreshold(confRange)
