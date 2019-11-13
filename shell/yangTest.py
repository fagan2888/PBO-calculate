
# edited by yangning

import pandas as pd
import pymysql
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

conn = pymysql.connect(host='192.168.88.254', user='yangning', passwd='yangning20d@11024950', database='multi_factor', charset='utf8')
sql_t = 'select * from rank_z_score_descriptor_return_1021'
df = pd.read_sql(con=conn, sql=sql_t, parse_dates=['trade_date'], index_col=['descriptor', 'trade_date']).sort_index()

descriptors = df.index.get_level_values(0).unique()
descriptor = 'earnings_to_price'
df_t = df.loc[descriptor].iloc[::10].copy()

ser_record = pd.Series(index=range(1000))
for i in range(1000):
    df_t2 = shuffle(df_t)
    df_train = df_t2.iloc[:int(df_t2.shape[0]/2)]
    df_test = df_t2.iloc[-int(df_t2.shape[0]/2):]
    ser_train = (df_train.mean() / df_train.std()).rank(ascending=False).sort_values()
    ser_test = (df_test.mean() / df_test.std()).rank(ascending=False).sort_values()
    #
    ser_record.at[i] = ser_test[ser_train.index[0]] / len(ser_test)

ser_record.plot(kind='kde')
ser_record[ser_record<0.5].count() / ser_record[ser_record<0.5].shape[0]
