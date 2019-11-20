#coding=utf8

from sqlalchemy import MetaData, Table, select, func, literal_column
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
from . import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# mz_markowitz
#
def load(gid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('mz_markowitz_alloc', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.mz_type,
        t1.c.mz_algo,
        t1.c.mz_markowitz_id,
        t1.c.mz_risk,
        t1.c.mz_name,
    ]

    s = select(columns)

    if gid is not None:
        s = s.where(t1.c.mz_markowitz_id == gid)

    df = pd.read_sql(s, db, index_col=['globalid'])

    return df

def where_markowitz_id(markowitz_id, risks=None, xtypes=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('mz_markowitz_alloc', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.mz_type,
        t1.c.mz_algo,
        t1.c.mz_risk,
        t1.c.mz_markowitz_id,
        t1.c.mz_name,
    ]

    s = select(columns)

    if markowitz_id is not None:
        s = s.where(t1.c.mz_markowitz_id == markowitz_id)
    if xtypes is not None:
        s = s.where(t1.c.mz_type.in_(xtypes))
    if risks is not None:
        s = s.where(t1.c.mz_risk.in_(risks))

    df = pd.read_sql(s, db)
    df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)

    return df

# def max_id_between(min_id, max_id):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t = Table('mz_markowitz', metadata, autoload=True)

#     columns = [ t.c.globalid ]

#     s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

#     return s.execute().scalar()

def save(gid, df):
    fmt_columns = ['mz_risk']
    fmt_precision = 2
    if not df.empty:
        df = database.number_format(df, fmt_columns, fmt_precision)
    #
    # 保存择时结果到数据库
    #
    db = database.connection('asset')
    t2 = Table('mz_markowitz_alloc', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.mz_markowitz_id == gid))
    df_old = pd.read_sql(s, db, index_col=['globalid'])
    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)

    # 更新数据库
    # print df_new.head()
    # print df_old.head()
    database.batch(db, t2, df, df_old, timestamp=True)

