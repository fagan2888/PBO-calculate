# edited by zhaoliyuan

# this file helps to calculate probability of backtest overffiting of an invest strategy

# CSCV (combinatorially symmetric cross-validation ) is used

import pandas as pd
import numpy as np
from sqlhelper import batch
from sqlalchemy.orm import sessionmaker
from db.tables import mz_markowitz_nav 
from itertools import combinations
import math
from sqlalchemy import *
from config import uris
import click
from ipdb import set_trace


# load data from asset database
def loadData():
    engine = batch.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()
    sql = session.query(mz_markowitz_nav.mz_markowitz_id, mz_markowitz_nav.mz_date, mz_markowitz_nav.mz_inc)
    sql = sql.filter(mz_markowitz_nav.mz_markowitz_id.like('MZ.00006%'))
    df = pd.read_sql(sql.statement, session.bind)
    session.commit()
    session.close()

    #  删除一些列，这些列的差距太明显，不适宜用作排序样本
    df = df[~df['mz_markowitz_id'].isin(['MZ.000060', 'MZ.000061','MZ.000062', 'MZ.000068', 'MZ.000069'])]
    df = df.set_index(['mz_date', 'mz_markowitz_id']).unstack()['mz_inc']

    newDf = pd.DataFrame({'date': list(df.index)})
    newDf.sort_values(by = 'date', ascending = True, inplace = True)
    k = 0
    for i in range(5):
        components = list(combinations(set(df.columns),i+1))
        for component in components:
            if len(component) == 1:
                new = list(df[component[0]])
            else:
                new = np.zeros(len(newDf))
                for name in component:
                    newadd = df[name]
                    new = list(np.array(new) + 1/(len(component))*np.array(newadd))

            newDf[str(k)] = new
            k += 1

#    newDf.to_csv('MZ.00006%.csv')
    return newDf

# cut origin data into S subsamples and make paris of IS and OOS
# calculate ranks
def calculate(df, column, S):
    df.sort_values(by = column, ascending = True, inplace = True)
    df.set_index(column, inplace = True)
    
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
        lamdas = lamdas + [lamda]
    
    lamdas.sort()

    return lamdas

# calculate PBO
# r 是一个评价标准，一定时间段下排名最好的，要求在其他时间段，其排名不得落后于r
# r越小标准越严格，PBO也越大
def PBOCal(lamdas, r):
    rank = 1 - r
    count = len([num for num in lamdas if num < math.log(rank/(1-rank))])
    PBO = count / len(lamdas)
    return PBO

@click.command()
@click.option('--r','optrank', help = 'criterion, small r means a more strict criterion', default = 0.5)
@click.option('--s','optslides',help = 'cut original data into several slides and make groups randomly', default = 16)
@click.pass_context
def cscv(ctx, optrank, optslides):
    df = loadData()
    if len(df) == 0:
        print('warinig! origin data was an empty dataframe')
        return -1
#    df = pd.read_csv('MZ.00006%.csv')
    column = 'date'
    if len(df) <= optslides:
        print('warning! number of slides is bigger than length of original data!')
        return -1
    lamdas = calculate(df, column, optslides)
    dfForSave = pd.DataFrame(lamdas, columns = ['lamda'])
    dfForSave.to_excel('lamda.xlsx')

    PBO = PBOCal(lamdas = lamdas, r = optrank)
    print('PBO is ',PBO)

    if PBO < 0.5:
        print ('it is very likely that this factor is useful')
    else:
        print('useless factor')

if __name__ =='__main__':
    cscv(obj={})

