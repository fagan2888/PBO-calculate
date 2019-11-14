# edited by zhaoliyuan

# this file helps to calculate probability of backtest overffiting of an invest strategy

# CSCV (combinatorially symmetric cross-validation ) is used

import pandas as pd
import numpy as np
from itertools import combinations
import math
from sqlalchemy import *
from config import uris
from ipdb import set_trace


def loadOriginData(base = 'multi_factor',table = 'rank_z_score_descriptor_return_1021', columns = '*', dates = ['trade_date'], index = ['descriptor', 'trade_date'], filters = 'where 1'):
    if base in uris.keys():
        uri = uris[base]
        conn = create_engine(uri)
    else:
        print('uri cannot found in config.py')
        return -1

    sql_t = 'select ' + columns + ' from '+ table + ' ' + filters
    df = pd.read_sql(con=conn, sql=sql_t, parse_dates = dates, index_col = index).sort_index()
    return df


# cut origin data into S subsamples and make paris of IS and OOS
# calculate ranks
def calculate(df, column, S):
    df.sort_values(by = column, ascending = True, inplace = True)
    
    signal = list()
    for i in range(0,S):
        signal.append(i)
    
    trainingDataNoGroup = list(set(combinations(signal, int(S/2))))
   
    lamdas = list()
    for trainingDataNo in trainingDataNoGroup:
        # for every case calculate the PBO
        trainData = pd.DataFrame(columns = list(df.columns))
        testData = pd.DataFrame(columns = list(df.columns))
        for i in range(S):
            sample = df.iloc[int(i*len(df)//S):int((i+1)*len(df)//S)]
            if i in trainingDataNo:
                trainData = pd.concat([trainData, sample], axis = 0)
            else:
                testData = pd.concat([testData, sample], axis = 0)
        # calculate
        trainDataSharpeRatio = trainData.mean() / trainData.std() 
        position = trainDataSharpeRatio.sort_values(ascending = False).rank().index[0]
        testDataSharpeRatio = testData.mean() / testData.std()
        rank = testDataSharpeRatio.rank()[position] / (len(testDataSharpeRatio)+1)
        lamda = math.log(rank/(1-rank))
        print('lamda = ', lamda)
        lamdas = lamdas + [lamda]
    
    lamdas.sort()

    return lamdas

# calculate PBO
def PBO(lamdas):
    count = len([num for num in lamdas if num < 0])
    PBO = count / len(lamdas)
    return PBO

df = loadOriginData(base = 'asset', table = 'mz_markowitz_nav', dates = ['mz_date'], index = ['mz_markowitz_id','mz_date'], filters = "where mz_markowitz_id = 'MZ.000060'")
print(df)
set_trace()
column = 'mz_date'
lamdas = calculate(df, column, 16)
dfForSave = pd.DataFrame(lamdas, columns = ['lamda'])
dfForSave.to_excel('lamda.xlsx')

PBO = PBO(lamdas)
print('PBO is ',PBO)

if PBO < 0.5:
    print ('it is very likely that this factor is useful')
else:
    print('useless factor')

