

# this file is to load basic data from database in order to do PBO calculate
# original matix is made here

import numpy as np
import pandas as pd
from sqlhelper import database
from sqlalchemy.orm import sessionmaker
from db.tables import mz_markowitz_nav 


def doer():
    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()
    sql = session.query(mz_markowitz_nav.mz_markowitz_id, mz_markowitz_nav.mz_inc)
    sql = sql.filter(mz_markowitz_nav.mz_markowitz_id.like('MZ.00006%'))
    df = pd.read_sql(sql.statement, session.bind, index_col = ['mz_markowitz_id'])
    session.commit()
    session.close()
    print(df)
    df.to_csv('MZ.00006%.csv')



if __name__ == '__main__':
    doer()


