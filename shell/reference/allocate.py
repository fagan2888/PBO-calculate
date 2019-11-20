#coding=utf8
'''
Modified on: May. 7, 2019
Editor: Shixun Su
Contact: sushixun@licaimofang.com
'''

import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import time
import logging
import re
import Const
import DFUtil
import DBData
import util_numpy as npu
import Portfolio as PF
from TimingWavelet import TimingWt
import multiprocessing
from multiprocessing import Manager
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, asset_mz_markowitz, asset_mz_markowitz_alloc, asset_mz_markowitz_argv,  asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe, asset_wt_filter_nav
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl, asset_allocate
from util import xdict
from util.xdebug import dd
from asset import Asset, WaveletAsset
from trade_date import ATradeDate


import traceback, code


logger = logging.getLogger(__name__)


class AssetBound(object):

    def __init__(self, globalid = None, asset_id = None, bound = None, upper = 0.7, start_date = None, end_date = None):

        self.__globalid = globalid
        self.__asset_id = asset_id
        self.__start_date = start_date
        self.__end_date = end_date
        #TODO : load from database by globalid

        if bound is None:
            asset_bound = {'sum1': 0, 'sum2' : 0, 'upper': upper, 'lower': 0.0, 'lower_sum1' : 0, 'lower_sum2' : 0, 'upper_sum1' : 0, 'upper_sum2' : 0, 'asset_id' : asset_id}
        else:
            asset_bound = bound
        trade_date = [pd.datetime(1900,1,1)]
        self.__bound = pd.DataFrame(asset_bound, index = trade_date)
        self.__bound.index.name = 'trade_date'

    @property
    def globalid(self):
        return self.__globalid

    @property
    def asset_id(self):
        return self.__asset_id.copy()

    @property
    def bound(self):
        return self.__bound.copy()

    @property
    def start_date(self):
        return self.__start_date

    @property
    def end_date(self):
        return self.__end_date


    def get_day_bound(self, day):
        bound = self.__bound.copy().sort_index(ascending=True)
        days = bound.index
        days = days[days <= day]
        last_day = days[-1]
        return bound.loc[last_day]


    @staticmethod
    def load_asset_bounds(globalid):
        df = asset_allocate.load_mz_markowitz_bounds(globalid)
        df = df.rename(columns = {'mz_upper_limit': 'upper', 'mz_lower_limit': 'lower', 'mz_sum1_limit': 'sum1', 'mz_sum2_limit': 'sum2',
                                'mz_lower_sum1_limit':'lower_sum1', 'mz_lower_sum2_limit':'lower_sum2', 'mz_upper_sum1_limit':'upper_sum1', 'mz_upper_sum1_limit':'upper_sum1',})
        bounds = {}
        for asset_id in df.index:
            bound_dict = df.loc[asset_id].to_dict()
            bound = bounds.setdefault(asset_id, [])
            bound.append(
                AssetBound(globalid = globalid, asset_id = asset_id, bound = bound_dict, start_date = bound_dict['mz_allocate_start_date'], end_date = bound_dict['mz_allocate_end_date'])
            )
        return bounds


    @staticmethod
    def get_asset_day_bound(asset_id, day, bounds):

        if type(day) == str:
            day = datetime.strptime(day, '%Y-%m-%d')

        if not (asset_id in bounds.keys()):
            return AssetBound(asset_id = asset_id).get_day_bound(day)
        else:
            for asset_bound in bounds[asset_id]:
                if pd.isnull(asset_bound.start_date) and pd.isnull(asset_bound.end_date):
                    return asset_bound.get_day_bound(day)
                elif (asset_bound.start_date <= day) and (pd.isnull(asset_bound.end_date) or (asset_bound.end_date > day)):
                    return asset_bound.get_day_bound(day)

            return AssetBound(asset_id = asset_id, upper = 0.0).get_day_bound(day)

    def to_db():
        pass



class Allocate(object):


    def __init__(self, globalid, assets = None, reindex = None, lookback = None, period = 1, bound = None):

        self.__globalid = globalid
        self.__assets = assets
        self.__pos = None
        self.__nav_sr = None
        self.__index = reindex
        self.__lookback = lookback
        self.__period = period
        self.__bound = {}
        if assets is not None:
            for asset_id in list(assets.keys()):
                self.__bound[asset_id] = [AssetBound('asset_bound_default', asset_id = asset_id)]
        if bound is not None:
            for asset_id in list(bound.keys()):
                self.__bound[asset_id] = bound[asset_id]


    @property
    def globalid(self):
        return self.__globalid

    @property
    def assets(self):
        return self.__assets

    @property
    def index(self):
        return self.__index

    @property
    def lookback(self):
        return self.__lookback

    @property
    def period(self):
        return self.__period

    @property
    def nav_sr(self):
        return self.__nav_sr

    @property
    def nav_df(self):
        if self.__nav_sr is None:
            return None
        else:
            nav_df = self.__nav_sr.to_frame()
            nav_df.columns = [self.__globalid]
        return nav_df

    @property
    def pos(self):
        return self.__pos

    @property
    def bound(self):
        return self.__bound

    def allocate(self):

        adjust_days = self.index[self.lookback::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        s = 'perform %-12s' % self.__class__.__name__

        with click.progressbar(
                adjust_days, label=s.ljust(30),
                item_show_func=lambda x:  x.strftime("%Y-%m-%d") if x else None) as bar:

            for day in bar:

                logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))
                df_inc, bound = self.load_allocate_data(day, asset_ids)

                ws = self.allocate_algo(day, df_inc, bound)

                for asset_id in list(ws.keys()):
                    pos_df.loc[day, asset_id] = ws[asset_id]

        # ser_days = pd.Series(adjust_days, index=adjust_days)
        # pos_df.loc[:, :] = ser_days.apply(self.allocate_pos_day)

        return pos_df

    def allocate_day(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)
        ws = self.allocate_algo(day)
        ws = pd.Series(ws)
        inc = np.dot(df_inc, ws)

        return ws, inc

    def load_allocate_data(self, day, asset_ids):

        reindex = self.index[self.index <= day][-self.lookback-1:]

        bound = []
        allocate_asset_ids = []
        for asset_id in asset_ids:
            asset_bound = AssetBound.get_asset_day_bound(asset_id, day, self.bound).to_dict()
            if asset_bound['upper'] > 0:
                bound.append(asset_bound)
                allocate_asset_ids.append(asset_id)

        data = {}
        for asset_id in allocate_asset_ids:
            data[asset_id] = self.assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc = df_nav.pct_change().iloc[1:]

        return df_inc, bound

    def allocate_algo(self, day, df_inc, bound):

        ws = [1.0 / len(df_inc.columns)] * len(df_inc.columns)

        ws = dict(list(zip(df_inc.columns.ravel(), ws)))

        return ws

    def nav_update(self):
        return

    def turnover_update(self):
        return

    def load_db(self):
        pass

    def to_db(self):
        pass


class AllocateNew(Allocate):

    def __init__(self, globalid, assets=None, reindex=None, lookback=None, period=1, bound=None):

        super(AllocateNew, self).__init__(globalid, assets, reindex, lookback, period, bound)

    def m_allocate(self):

        adjust_days = self.index[self.lookback::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index=adjust_days, columns=asset_ids)

        # for day in adjust_days:
            # pos_df.loc[day, :] = self.allocate_algo(day)

        cpu_count = multiprocessing.cpu_count()//2 if multiprocessing.cpu_count()//2 <= 32 else 32

        pool = multiprocessing.Pool(cpu_count)
        pos = pool.map(self.allocate_algo, adjust_days.to_list())
        pool.close()
        pool.join()

        pos_df = pd.DataFrame(pos, index=adjust_days).fillna(0.0)

        return pos_df

    def allocate_pos_day(self, day):

        asset_ids = list(self.assets.keys())
        df_inc, bound = self.load_allocate_data(day, asset_ids)
        pos = self.allocate_algo(day, df_inc, bound).rename(day)

        return pos

    def load_allocate_data(self, day, asset_ids, data_period='week'):

        bound = []
        allocate_asset_ids = []
        for asset_id in asset_ids:
            asset_bound = AssetBound.get_asset_day_bound(asset_id, day, self.bound).to_dict()
            if asset_bound['upper'] > 0:
                bound.append(asset_bound)
                allocate_asset_ids.append(asset_id)

        reindex = self.index[self.index <= day][-self.lookback-1:]
        if data_period == 'week':
            pass
        elif data_period == 'day':
            reindex = ATradeDate.trade_date(begin_date=reindex[0], end_date=day)
        else:
            raise ValueError

        data = {}
        for asset_id in allocate_asset_ids:
            data[asset_id] = self.assets[asset_id].nav(reindex=reindex)

        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc = df_nav.pct_change().iloc[1:]

        return df_nav, df_inc, bound


if __name__ == '__main__':

    asset = Asset('120000001')
    print(asset.nav().tail())

    asset = WaveletAsset('120000013', 2)
    print(asset.nav('1900-01-01', datetime.now()).tail())

    trade_date = ATradeDate.week_trade_date(begin_date = '2012-01-01')
    print(trade_date[-5:])

    asset_globalids = ['120000001', '120000002', '120000013', '120000014', '120000015']
    assets = {}
    for asset_id in asset_globalids:
        assets[asset_id] = Asset(asset_id)

    #allocate = Allocate('ALC.000001', assets, trade_date, 26)
    #print(allocate.allocate())

    bounds = AssetBound.load_asset_bounds('AB.000001')
    print(AssetBound.get_asset_day_bound('120000016', '2003-01-03', bounds))
    #print(bound.get_day_bound('2018-01-01'))
