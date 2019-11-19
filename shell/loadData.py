

# this file is to load basic data from database in order to do PBO calculate
# original matix is made here

import numpy as np
import pandas as pd
from sqlhelper import database
from sqlalchemy.orm import sessionmaker
from db.tables import mz_markowitz_nav 
from itertools import combinations
from ipdb import set_trace

def doer():
    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()
    sql = session.query(mz_markowitz_nav.mz_markowitz_id, mz_markowitz_nav.mz_date, mz_markowitz_nav.mz_inc)
    sql = sql.filter(mz_markowitz_nav.mz_markowitz_id.like('MZ.00006%'))
    df = pd.read_sql(sql.statement, session.bind)
    session.commit()
    session.close()
    df['mz_date'] = df['mz_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    df = df.sort_values(by = 'mz_date')
    date = set(df[df['mz_markowitz_id']=='MZ.000060']['mz_date'])
    for i in range(1,10):
        date = date.intersection(set(df[df['mz_markowitz_id'] == 'MZ.00006'+ str(i)]['mz_date']))
    df = df[df['mz_date'].isin(date)]
    newDf = pd.DataFrame({'date': list(date)})
    newDf.sort_values(by = 'date', ascending = True, inplace = True)
    k = 0
    for i in range(5):
        components = list(combinations(set(df['mz_markowitz_id']),i+1))
        for component in components:
            if len(component) == 1:
                dfInc = df.loc[df.mz_markowitz_id.isin(component)].sort_values(by = 'mz_date', ascending = True)
                new = list(dfInc['mz_inc'])
            else:
                new = np.zeros(len(date))
                for name in component:
                    dfInc = df.loc[df.mz_markowitz_id == name].sort_values(by = 'mz_date', ascending = True)
                    new = list(np.array(new) + 1/len(component)*np.array(dfInc['mz_inc']))

            newDf[str(k)] = new
            k += 1

    newDf.set_index('date', inplace = True)
    print(newDf)
    set_trace()
    newDf.to_csv('MZ.00006%.csv')



if __name__ == '__main__':
    doer()

