import numpy    as np
import pandas   as pd

from core.nn.loss import l1_norm_error

# =============================================== COMPARE SEQUENCE =====================================================
def compare_sequence(source, candidate, errorFunc):
    '''
    Compare 2 countries growth similarity
    :param source:      data for source country
    :param candidate:   data for candidate country
    :param errorFunc:   callable error
    :return: the minimum error and the index of where it was computed
    '''

    minError = np.inf
    minIdx   = -1
    # only check the countries that can influence
    if len(candidate) > len(source):
        noWindows  = len(candidate) - len(source)
        windowSize = len(source)

        # sliding window over candidate country
        for i in range(0, noWindows):

            # compute the loss
            error = errorFunc(source, candidate[i:i + windowSize])

            # save the min error
            if error <= minError:
                minError = error
                minIdx   = i

        return minError, minIdx

    return None, None

# =============================================== GET NEAREST SEQUENCE =================================================
def get_nearest_sequence(df, state, alignThreshConf=50, alignThreshDead=10, errorFunc=l1_norm_error):
    '''
    :param df:                  df containing all countries and states
    :param state:               target state
    :param alignThreshConf:     minimum number of confirmed cases
    :param alignThreshDead:     minimum number of fatalities
    :param errorFunc:           error to be applied
    :return: dataframe containing the the error and the align index for each candidate country 
    '''
    resDf  = pd.DataFrame(columns=['Province_State', 'deathError', 'confirmedError', 'deathIdx', 'confirmedIdx'])

    confDf = df[df['ConfirmedCases'] > alignThreshConf]
    deadDf = df[df['Fatalities'] > alignThreshDead]

    # merge provinces
    if state in ["China", "Australia"]:
        df = df[df["Date"] >= pd.to_datetime("2020-03-01")]
        confDf = df[df['ConfirmedCases'] > alignThreshConf]
        deadDf = df[df['Fatalities'] > alignThreshDead]
        to_merge = confDf[confDf['Country_Region'] == state]
        dead_to_merge = deadDf[deadDf["Country_Region"] == state]
    else:
        to_merge = confDf[confDf['Province_State'] == state]
        dead_to_merge = deadDf[deadDf["Province_State"] == state]

    if len((to_merge["Province_State"].unique())) > 1:
        to_merge = to_merge.groupby("Date").mean().reset_index()
        to_merge["Province_State"] = state
        to_merge["Country_Region"] = state

        dead_to_merge = dead_to_merge.groupby("Date").mean().reset_index()
        dead_to_merge["Province_State"] = state
        dead_to_merge["Country_Region"] = state

    # get source region data
    regionDfConf = to_merge.sort_values(by='Date', ascending=True)
    regionDfDead = dead_to_merge.sort_values(by='Date', ascending=True)
    
    regionConf = regionDfConf['ConfirmedCases'].values
    regionDead = regionDfDead['Fatalities'].values

    # check all possible candidates
    for neighbour in df['Country_Region'].unique():

        # skip comparing with the same country
        if neighbour == state:
            continue

        # get candidate country
        confNeighDf = confDf[confDf['Country_Region'] == neighbour].sort_values(by='Date', ascending=True)
        deadNeighDf = deadDf[deadDf['Country_Region'] == neighbour].sort_values(by='Date', ascending=True)

        if len((confNeighDf["Province_State"].unique())) > 1:
            confNeighDf = confNeighDf.groupby("Date").mean().reset_index()
            deadNeighDf = deadNeighDf.groupby("Date").mean().reset_index()

        neighConf = confNeighDf['ConfirmedCases'].values
        neighDead = deadNeighDf['Fatalities'].values
        
        # get error for confirmed and fatalities
        confErr, confIdx = compare_sequence(regionConf, neighConf, errorFunc)
        deadErr, deadIdx = compare_sequence(regionDead, neighDead, errorFunc)

        # the candidate will be ignored if it does not have enough data
        if confErr is None:
            continue

        # append result
        res = {'Province_State': neighbour, 'deathError': deadErr, 'confirmedError': confErr,
               'deathIdx': deadIdx, 'confirmedIdx': confIdx}

        resDf = resDf.append(res, ignore_index=True)


    return resDf
