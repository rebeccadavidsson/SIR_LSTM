import os
from core.nn.LSTMmodel import LSTM
from SIR_ODE import SIR
import math
import pickle
import datetime
from numpy import array
import pandas as pd
import covsirphy as cs
import requests, io, json, urllib
import numpy as np
from itertools import cycle
import os.path
import math
import seaborn as sns
sns.set()


def SA(COUNTRY, PARAMS, NPI_df, jhu_data, population_data):

    ThreshConf   = 70
    ThreshDead   = 20
    TARGET       = "New Confirmed"
    TYPE         = "LSTMCell"
    FUTURE_DAYS  = 5
    DELAY_START  = 16
    # TRAIN_UP_TO  = pd.to_datetime("2020-12-01")

    # lstm = LSTM(COUNTRY, TRAIN_UP_TO, FUTURE_DAYS, ThreshDead, TARGET, TYPE, DELAY_START, PARAMS)
    # lstm.simulate(ThreshConf=70)

    errors = pd.DataFrame(columns = ["RMSE", "MAE", "MAPE"])
    days_delay = PARAMS["days_delay"]
    TARGET_NPI = PARAMS["TARGET"]

    NPI_df = NPI_df[NPI_df["Country"] == COUNTRY]

    # Get lockdown dates
    NPI_df = NPI_df.reset_index().drop('index', axis=1)
    NPI_df = NPI_df.groupby("Date").mean().reset_index()

    NPI = TARGET_NPI
    NPI_dates = {}

    lockdown_indexes = NPI_df[NPI_df[NPI] >= 75].index
    lockdown_dates = NPI_df[NPI_df[NPI] >= 75]["Date"]
    periods = get_periods(lockdown_indexes)

    if periods == []:
        lockdown_indexes = NPI_df[NPI_df[NPI] >= 65].index
        lockdown_dates = NPI_df[NPI_df[NPI] >= 65]["Date"]
        periods = get_periods(lockdown_indexes)

    if periods == []:
        lockdown_indexes = NPI_df[NPI_df[NPI] >= 4].index
        lockdown_dates = NPI_df[NPI_df[NPI] >= 4]["Date"]
        periods = get_periods(lockdown_indexes)

    if periods == []:
        lockdown_indexes = NPI_df[NPI_df[NPI] >= 3].index
        lockdown_dates = NPI_df[NPI_df[NPI] >= 3]["Date"]
        periods = get_periods(lockdown_indexes)

    lockdown_dates_adjusted = []
    for date in lockdown_dates:
        new_date = date + datetime.timedelta(days = days_delay)
        lockdown_dates_adjusted.append(new_date)
    NPI_dates[NPI] = lockdown_dates_adjusted

    lockdown_dates_adjusted = []
    for date in lockdown_dates:
        new_date = date + datetime.timedelta(days = days_delay)
        lockdown_dates_adjusted.append(new_date)
    lockdown_dates_adjusted = pd.Series(lockdown_dates_adjusted) 

    # Save variable DELAY_START, which is equal to the start of a lockdown period
    if len(periods) < 1:
        return False
    DELAY_START = NPI_df.iloc[periods[len(periods) - 1][0]].Date

    df = jhu_data
    df = df[(df["Country"] == COUNTRY) & (df["Province"] == "-")]
    df["New Confirmed"] = df["Confirmed"].diff()

    if COUNTRY == "United Kingdom":
        df_params = pd.read_pickle("./data/df_United_Kingdom")
    if COUNTRY in ["Australia", "China", "Japan", "United States"]:
        df_params = pd.read_pickle("./data/df_Chi_Aus_Jap_Us")
    elif COUNTRY == "Sweden":
        df_params = pd.read_pickle("../figures/pickles/df_Sweden")
    elif COUNTRY == "United States":
        df_params = pd.read_pickle("../figures/pickles/df_United_States")
    else:
        df_params = pd.read_pickle("../figures/pickles/df_9_countries")

    df_params = df_params[df_params["Country"] == COUNTRY]

    TRAIN_UP_TO  = DELAY_START 

    country_df = jhu_data
    country_df = country_df[(country_df["Country"] == COUNTRY) & (country_df["Province"] == "-") ]
    selection = country_df[country_df["Date"] == DELAY_START + datetime.timedelta(days_delay + 2)]

    selection["Confirmed"] = abs(selection["Confirmed"].values[0] - selection["Confirmed"].values[0])
    population_df = population_data
    N = population_df[population_df["Country"] == COUNTRY]["Population"].values[0]
    selection.head()

    target_column = "Confirmed"
    if selection["Confirmed"].values[0] == 0:
        target_column = "Infected"

    def calc_param(df, lockdown_dates):
        total_params = ["theta", "kappa", "rho", "sigma"]
        calc_params_df = {}
        for param in total_params:
            values = []
            for date in df["Date"].values:
                if date in lockdown_dates.values:
                    values.append(np.mean(df[df['Date'] == date][param]))
            calc_params_df[param] = np.mean(values)
        return calc_params_df

    # params = calc_param(df_params, lockdown_dates_adjusted)
    params_total = {}
    sir_params_total = {}

    for p in NPI_dates:
        res = calc_param(df_params, pd.Series(NPI_dates[p]))
        if not math.isnan(res["kappa"]):
            params_total[p] = res

            sir = SIR(N=N, I0=selection[target_column].values[0], R0=selection["Recovered"].values[0], 
                      beta=res["rho"], gamma=res["theta"],
                     days=85)
            SIR_results = sir.simulate(target="Infected")
            sir_params_total[p] = SIR_results

    # -------------------------- LSTM -------------------------#
    ThreshConf   = 70
    ThreshDead   = 20
    TARGET       = "New Confirmed"
    TYPE         = "LSTMCell"
    FUTURE_DAYS  = 5
    RUNS         = 1
    ERROR_THRESH = 1.7
    WITH_BIAS    = True

    BIAS = sir_params_total[TARGET_NPI]

    fname = f"results/res_{TARGET_NPI}_{WITH_BIAS}_{COUNTRY}.p"
    results_df = pd.DataFrame(columns=["Date"])

    # Check if dataframe already exists to build up on
    if os.path.isfile(fname):
        results_df = pickle.load(open( fname, "rb" ))
        j = len(results_df.columns)
        RUNS = j + RUNS
    else:
        j = 0

    while j < RUNS:
        print("RUN", j)
        FUTURE_DAYS = 5
        lstm = LSTM(COUNTRY, TRAIN_UP_TO, FUTURE_DAYS, ThreshDead, TARGET, TYPE, DELAY_START, PARAMS, N)
        results1 = lstm.simulate(ThreshConf=70)
        bias_results = add_bias(results1, BIAS, DELAY_START, days_delay, SIR_results)

        for i in range(2):
            FUTURE_DAYS += 6
            lstm = LSTM(COUNTRY, TRAIN_UP_TO, FUTURE_DAYS, ThreshDead, TARGET, TYPE, DELAY_START, PARAMS, N)
            results2 = lstm.simulate(ThreshConf=70, input_data=bias_results)
            bias_results = add_bias(results2, BIAS,  DELAY_START, days_delay, SIR_results, isBias=WITH_BIAS)

        trainData = pd.DataFrame(results2["trainData"])
        new_col = pd.DataFrame(trainData)
        new_col = new_col.set_index("Date")

        if j == 0:
            valDates = pd.Series(results2["valData"]["Date"])
            trainDates = pd.Series(results2["trainData"]["Date"])
            new_dates = trainDates.append(valDates)
            results_df["Date"] = new_dates
            results_df = results_df.set_index("Date")
            trueCases = results1["trainData"].set_index("Date")
            valCases = results1["valData"].set_index("Date")
            results_df["TrueCases"] = trueCases["ConfirmedCases"]
            results_df["valCases"] = valCases["ConfirmedCases"]

        error = results2["error"]
    

        # Don't save run if there were errors in prediction
        if error > ERROR_THRESH:
            # print(error)
            RUNS += 1
        else:
            if j == 0:
                results_df["ConfirmedCases"] = new_col["ConfirmedCases"]
            else:
                new_col = new_col.rename(columns={"ConfirmedCases": "Cases" + str(j)})
                results_df = pd.concat([results_df, new_col["Cases" + str(j)]], axis=1)
            err = results2["errors"]
            errors = errors.append(pd.Series(results2["errors"], index=errors.columns ), ignore_index=True)
        j += 1

    return errors


def get_periods(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))
    
def add_bias(results, BIAS, DELAY_START, days_delay, SIR_results,  isBias=True):
    data = results.copy()
    preds = []
    SIR_data = []
    dates = []
    x_dates, x_preds, total_dates = [], [], []
    
    # START Adding bias after days_delay
    BIAS_START = DELAY_START + datetime.timedelta(days_delay)
    for date, pred, index in zip(data["valData"]["Date"], data["pred"], range(0, len(data["valData"]))):
        if date >= BIAS_START:
            preds.append(pred)
            dates.append(date)
            SIR_data.append(SIR_results["I"][index])
        x_dates.append(date)
        x_preds.append(pred)
        total_dates.append(date)
     # Add bias to prediction
    # Calculate trend in SIR-predictions
    x = np.arange(0,len(SIR_data))
    y = np.array(SIR_data)
    if len(y) > 1:
        z = np.polyfit(x, y, 1)[0]
    else:
        z = 1
        
    new_preds, old_preds = [], []
    weight = 3
    for i in range(len(preds)):
        if i == len(preds) - 1:
            diff = preds[i] - preds[i-1]
        else:
            diff = preds[i + 1] - preds[i]
        percent = (i+1) / days_delay
        weight += 0.1
        if isBias:
            new_trend = (percent * z/weight + diff)
            new_trend = new_trend + diff
            new_preds.append(preds[i] + new_trend)
        else:
            new_preds.append(preds[i])
        old_preds.append(preds[i])

    if len(y) <= 1:
        new_preds = data["pred"]
    combined_new_preds = new_preds

    # Add data to results
    data["total_old_pred"] = data["pred"]
    data["pred"] = combined_new_preds
    data["oldpred"] = old_preds
    return data
