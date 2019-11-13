# edited by zhaoliyuan

# this file helps to calculate probability of backtest overffiting of an invest strategy

# CSCV (combinatorially symmetric cross-validation ) is used

import pandas as pd
import numpy as np
from itertools import combinations
import math
from sqlalchemy import *
from config import uris

key = 'multi_factor'
def loadOriginData():
    if key in uris.keys():
        uri = uris[key]
        conn = create_engine(uri)
    else:
        print('uri cannot found in config.py')
        return -1

    sql_t = 'select * from rank_z_score_descriptor_return_1021'
    df = pd.read_sql(con=conn, sql=sql_t, parse_dates=['trade_date'], index_col=['descriptor', 'trade_date']).sort_index()
    return df

df = loadOriginData()
# cut origin data into S subsamples and make paris of IS and OOS
# calculate ranks
def calculate(df, column, S):
    df.sort_values(by = column, ascending = True, inplace = True)
    
    signal = '1'
    for i in range(1,S):
        signal = signal + str(i+1)
    
    trainingDataNoGroup = list(set(combinations(signal, int(S/2))))
    for trainingDataNo in trainingDataNoGroup:
        # for every case calculate the PBO
        trainData = pd.DataFrame(columns = list(df.columns))
        testData = pd.DataFrame(columns = list(df.columns))
        lamdas = list()
        for i in range(S):
            sample = df.iloc[int(i*len(df)//S):int((i+1)*len(df)//S)]
            if str(i) in trainingDataNo:
                trainData = pd.concat([trainData, sample])
            else:
                testData = pd.concat([testData, sample])
        # calculate
        trainDataSharpeRatio = trainData.mean() / trainData.std() 
        position = trainDataSharpeRatio.sort_values(ascending = False).rank().index[0]
        testDataSharpeRatio = testData.mean() / testData.std()
        rank = testDataSharpeRatio.rank()[position] / (len(testDataSharpeRatio)+1)
        lamda = math.log(rank/(1-rank))
        lamdas = lamdas.append(lamda)
    
    lamdas.sort()

    return lamdas

column = 'trade_date'
lamdas = calculate(df, column, 16)
dfForSave = pd.DataFrame(lamdas, columns = ['lamda'])
dfForSave.to_excel('lamda.xlsx')
