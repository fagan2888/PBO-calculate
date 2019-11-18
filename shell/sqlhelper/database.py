
import sys
sys.path.append('../shell')
import string
import config
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import logging
import re

from sqlalchemy import *
from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy.engine import Engine as engine
from sqlalchemy.pool import Pool
from ipdb import set_trace

from dateutil.parser import parse

import MySQLdb
from config import uris
import os
from collections import Iterable

logger = logging.getLogger(__name__)

connections = {}

def connection(key):
    if key in connections:
        return connections[key]

    if key in uris:
        uri = uris[key]
        con = create_engine(uri)
        connections[key] = con
        return con

    return None


@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    connection_record.info['pid'] = os.getpid()


@event.listens_for(engine, "checkout")
def checkout(dbapi_connection, connection_record, connection_proxy):
    pid = os.getpid()
    if connection_record.info['pid'] != pid:
        connection_record.connection = connection_proxy.connection = None
        raise exc.DisconnectionError(
            "Connection record belongs to pid %s, "
            "attempting to check out in pid %s" %
            (connection_record.info['pid'], pid)
        )

@event.listens_for(Pool, "checkout")
def ping_connection(dbapi_connection, connection_record, connection_proxy):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SELECT 1")
    except:
        # optional - dispose the whole pool
        # instead of invalidating one at a time
        # connection_proxy._pool.dispose()
        # raise DisconnectionError - pool will try
        # connecting again up to three times before raising.
        raise exc.DisconnectionError()
        cursor.close()


def format(df, columns=[], fmter=None, kwcolumns=[]):
    if columns and fmter:
        for column in columns:
            if column in df.columns and column not in kwcolumns:
                df[column] = df[column].map(fmter)

    if kwcolumns:
        for k, v in kwcolumns:
            if v and k in df.columns:
                df[k] = df[k].map(v)
    return df

def number_format(df, columns=[], precision=2, **kwcolumns):
    if columns:
        for column in columns:
            if column in df.columns and column not in kwcolumns:
                df[column] = df[column].map(("{:.%df}" % (precision)).format)

    if kwcolumns:
        for k, v in kwcolumns.items():
            if v and k in df.columns:
                df[k] = df[k].map(("{:.%df}" % (v)).format)
    return df

def batch(dbname, tablename, df_new, df_old, delete=True, timestamp=True):
    db = connection(dbname)
    table = Table(tablename, MetaData(bind = db), autoload = True)
    try:
        db = db.connect()
        table = Table(table.name, MetaData(bind=db), autoload=True)
        transaction = db.begin()
        
        index_insert = df_new.index.difference(df_old.index)
        if delete:
            index_delete = df_old.index.difference(df_new.index)
        else:
            index_delete = tuple()
        index_update = df_new.index.intersection(df_old.index)
        
        if len(index_update):
            df1 = df_new.loc[index_update].copy()
            df2 = df_old.loc[index_update].copy()
            masks = (df1 != df2)
            df_update = df1.loc[masks.any(axis=1)].copy()
        else:
            df_update = pd.DataFrame(columns=df_new.columns)
        
        if len(df_update) > 50:
            keys = [table.c.get(c) for c in df_old.index.names]
            for segment in chunks(df_update.index.tolist(), 500):
                if len(df_old.index.names) == 1:
                    s = table.delete(keys[0].in_(segment))
                else:
                    segment = [tuple(map(str,eachTuple)) for eachTuple in segment]
                    s = table.delete(tuple_(*keys).in_(segment))
                s.execute()
            
            if timestamp:
                df_update['updated_at'] = df_update['created_at'] = datetime.now()
            
            df_update.to_sql(table.name, db, index=True, if_exists='append', chunksize=500)
             
            if len(df_update.index) >= 1:
                logger.info("delete %s (%5d records) and reinsert (%5d records): %s " % (table.name, len(df_update.index), len(df_update.index), df_update.index[0]))
        else:            
            if len(df_update):
                for key, row in df_update.iterrows():
                    origin = df2.loc[key]
                    columns = row[masks.loc[key]]

                    pkeys = []
                    if len(df_update.index.names) == 1:
                        pkeys.append((df_update.index.names[0], key))
                    else:
                        pkeys = list(zip(df_update.index.names, key))

                    pkeys = [tuple(map(str,eachTuple)) for eachTuple in pkeys]
                    dirty = {k:{'old':origin[k], 'new':v} for k,v in columns.items()}

                    if timestamp:
                        columns['updated_at'] = text('NOW()')

                    values = {k:v for k,v in columns.items()}
                    stmt = table.update(values=values)
                    for k, v in pkeys:
                        stmt = stmt.where(table.c.get(k) == v)

                    logger.info("update %s : {key: %s, dirties: %s} " % (table.name, str(pkeys), str(dirty)))
                    stmt.execute()
                    
        if len(index_delete):
            keys = [table.c.get(c) for c in df_new.index.names]
            for segment in chunks(index_delete.tolist(), 500):
                if len(df_new.index.names) == 1:
                    s = table.delete(keys[0].in_(segment))
                else:
                    segment = [tuple(map(str,eachTuple)) for eachTuple in segment]
                    s = table.delete(tuple_(*keys).in_(segment))
                s.execute()

            if len(index_delete) > 1:
                logger.info("delete %s (%5d records) : %s " % (table.name, len(index_delete), index_delete[0]))
    
        if len(index_insert):
            df_insert = df_new.loc[index_insert].copy()
            if timestamp:
                df_insert['updated_at'] = df_insert['created_at'] = datetime.now()

            df_insert.to_sql(table.name, db, index=True, if_exists='append', chunksize=500)
            
            if len(index_insert) > 1:
                logger.info("insert %s (%5d records) : %s " % (table.name, len(index_insert), index_insert[0]))

    except Exception as e:
        print(e)
        transaction.commit()
    finally:
        transaction.commit()


