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
import scipy
import scipy.optimize
import time
import logging
import re
import Const
import DFUtil
import DBData
import util_numpy as npu
import Portfolio as PF
import Financial as fin
from TimingWavelet import TimingWt
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Pool
from functools import partial
import random
from ipdb import set_trace
from sklearn.covariance import empirical_covariance, ledoit_wolf

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, asset_mz_markowitz, asset_mz_markowitz_alloc, asset_mz_markowitz_argv,  asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe, asset_wt_filter_nav
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl, asset_stock, asset_ra_bl_view
from db.asset_stock_factor import *
from util import xdict
from util.xdebug import dd
from asset import Asset, WaveletAsset
from allocate import Allocate, AllocateNew, AssetBound
from trade_date import ATradeDate
from view import View
import Financial as fin
import RiskParity
import util_optimize
from resample_tools import GaussianCopula

import PureFactor
import IndexFactor
import traceback, code
from monetary_fund_filter import MonetaryFundFilter


logger = logging.getLogger(__name__)


class AvgAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(AvgAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)

    def allocate_algo(self, day, df_inc, bound):
        ws = [1.0 / len(df_inc.columns)] * len(df_inc.columns)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class MzAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MzAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)

    def allocate_algo(self, day, df_inc, bound):
        risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


'''new'''
class RpAllocate(AllocateNew):

    def __init__(self, globalid, assets, reindex, lookback, period=1, bound=None):

        super(RpAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)

        asset_num = df_inc.shape[1]
        V = df_inc.cov()
        x_t = np.array([1 / asset_num] * asset_num)

        ws = RiskParity.cal_weight(V, x_t)
        ws = pd.Series(ws, index=df_inc.columns)

        return ws


'''new'''
class MzBlAllocate(AllocateNew):

    def __init__(self, globalid, assets, views, reindex, lookback, period=1, bound=None):

        super(MzBlAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self._views = views

    @property
    def views(self):

        return self._views

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)

        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        ws = self.markowitz_r_spe_bl(df_inc, P, eta, alpha, bound)
        ws = pd.Series(ws, index=df_inc.columns, name=day)

        return ws

    def load_bl_view(self, day, asset_ids):

        confidences = []
        view = pd.Series(0, index = asset_ids)
        for asset_id in asset_ids:
            view.loc[asset_id] = self.views[asset_id].view(day)
            confidences.append(self.views[asset_id].confidence)

        eta = np.array(abs(view[view!=0]))
        P = np.diag(np.sign(view))
        P = np.array([i for i in P if i.sum()!=0])
        if eta.size == 0:
            P = None
            eta = np.array([])

        return P, eta, np.mean(confidences)

    def markowitz_r_spe_bl(self, funddfr, P, eta, alpha, bounds):

        final_risk = 0
        final_return = 0
        final_ws = []
        final_ws = list(1.0 * np.ones(len(funddfr.columns)) / len(funddfr.columns))
        final_sharpe = -np.inf
        final_codes = []

        codes = funddfr.columns
        return_rate = []
        for code in codes:
            return_rate.append(funddfr[code].values)

        if eta.size == 0:
            risk, ret, ws = fin.efficient_frontier_spe(return_rate, bounds)
        else:
            risk, ret, ws = fin.efficient_frontier_spe_bl(return_rate, P, eta, alpha, bounds)

        for j in range(len(risk)):
            if risks[j] == 0:
                continue
            sharpe = (ret[j] - Const.rf) / risk[j]
            if sharpe > final_sharp:
                final_risk = risk[j]
                final_return = ret[j]
                final_ws = ws[j]
                final_sharpe = sharpe

        # final_res = np.append(final_ws, np.array([final_risk, final_ret, final_sharpe]))

        return final_ws


'''old'''
class MzBootAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period=1, bound=None, cpu_count=None, bootstrap_count=0):

        super(MzBootAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self._cpu_count = cpu_count
        else:
            self._cpu_count = cpu_count
        self._bootstrap_count = bootstrap_count

    def allocate_algo(self, day, df_inc, bound):

        risk, returns, ws, sharpe = PF.markowitz_bootstrap(df_inc, bound, cpu_count = self._cpu_count, bootstrap_count = self._bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))

        return ws


'''new'''
class MzBootBlAllocate(MzBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, period=1, bound=None, bootstrap_count=0):

        super(MzBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, bound)

        self._bootstrap_count = bootstrap_count

    @property
    def bootstrap_count(self):

        return self._bootstrap_count

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)

        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        ws = self.markowitz_bootstrap_bl(day, df_inc, P, eta, alpha, bound, self.bootstrap_count)

        return ws

    def markowitz_bootstrap_bl(self, day, df_inc, P, eta, alpha, bound, bootstrap_count):

        look_back = len(df_inc)
        loop_num = self.get_loop_num(bootstrap_count, look_back)

        day_indices = self.create_day_indices(look_back, loop_num)

        args = (df_inc, P, eta, alpha, bound)
        v_markowitz_random_bl = np.vectorize(partial(self.markowitz_random_bl, *args), signature='(n)->(m)')
        ws = np.mean(v_markowitz_random_bl(day_indices), axis=0)
        ws = pd.Series(ws, index=df_inc.columns, name=day)

        return ws

    def get_cpu_count(self, cpu_count):

        if cpu_count is None or cpu_count <= 0:
            return max(int(multiprocessing.cpu_count()) // 2, 1)
        else:
            return min(cpu_count, multiprocessing.cpu_count())

    def get_loop_num(self, bootstrap_count, look_back):

        if bootstrap_count is None or bootstrap_count <= 0:
            return look_back * 4
        elif bootstrap_count % 2:
            return bootstrap_count + 1
        else:
            return bootstrap_count

    def create_day_indices(self, look_back, loop_num):

        rep_num = loop_num * (look_back // 2) // look_back

        day_indices = list(range(0, look_back)) * rep_num
        random.shuffle(day_indices)

        day_indices = np.array(day_indices)
        day_indices = day_indices.reshape(len(day_indices) // (look_back // 2), look_back // 2)

        return day_indices

    def markowitz_random_bl(self, df_inc, P, eta, alpha, bound, random_index):

        tmp_df_inc = df_inc.iloc[random_index]
        ws = self.markowitz_r_spe_bl(tmp_df_inc, P, eta, alpha, bound)

        return ws

    # def unpack_res(self, res, keys):

        # risk = res[-3]
        # ret = res[-2]
        # ws = dict(zip(keys, res[:-3]))
        # sharpe = res[-1]

        # return risk, ret, ws, sharpe


class MzBootDownRiskAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzBootDownRiskAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self._cpu_count = cpu_count
        else:
            self._cpu_count = cpu_count
        self._bootstrap_count = bootstrap_count


    def allocate_algo(self, day, df_inc, bound):
        df_inc[df_inc >= 0] = 0.0
        risk, returns, ws, sharpe = self.markowitz_bootstrap(df_inc, bound, cpu_count = self._cpu_count, bootstrap_count = self._bootstrap_count)
        tdate = ATradeDate.trade_date()
        var = np.array([self.assets[code].origin_nav_sr.reindex(tdate).pct_change().loc[df_inc.index[-13]:df_inc.index[-1]].var() for code in df_inc.columns])
        ws = np.array(ws).ravel()
        ws = ws/var
        ws = ws/ws.sum()
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class FactorValidAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):

        super(FactorValidAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, stock_ids = assets.keys(), begin_date = '2010-01-01')
        self.sfr = load_stock_factor_return(sf_ids = sf_ids, begin_date = '2012-01-01')

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        pool = Pool(32)
        wss = pool.map(self.allocate_algo, adjust_days)
        pool.close()
        pool.join()

        for day, ws in zip(adjust_days, wss):
            for asset_id in ws.keys():
                pos_df.loc[day, asset_id] = ws[asset_id]

        return pos_df


    def allocate_algo(self, day):

        begin_date = (day.date() - timedelta(self.lookback - 1)).strftime('%Y-%m-%d')
        end_date = day.date().strftime('%Y-%m-%d')

        sfe = self.sfe[(self.sfe.index.get_level_values(2) > begin_date) & (self.sfe.index.get_level_values(2) < end_date)].reset_index()
        sfr = self.sfr[(self.sfr.index.get_level_values(1) > begin_date) & (self.sfr.index.get_level_values(1) < end_date)].reset_index()

        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)
        sfe = sfe[self.sf_ids]

        sfr = sfr.set_index(['trade_date', 'sf_id'])
        sfr = sfr.unstack()
        sfr.columns = sfr.columns.droplevel(0)
        sfr = sfr[self.sf_ids]

        R = sfr.mean()
        factor_weights = np.sign(R.values) * 0.5

        stock_weights = PureFactor.cal_weight(sfe, factor_weights)
        ws = dict(zip(sfe.index, stock_weights))

        return ws


'''old'''
class MzFixRiskBootAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzFixRiskBootAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self._cpu_count = cpu_count
        else:
            self._cpu_count = cpu_count
        self._bootstrap_count = bootstrap_count
        self.risk = risk

    def allocate_algo(self, day, df_inc, bound):
        risk, returns, ws, sharpe = PF.markowitz_bootstrap_fixrisk(df_inc, bound, self.risk, cpu_count = self._cpu_count, bootstrap_count = self._bootstrap_count)
        # risk, returns, ws, sharpe = PF.markowitz_fixrisk(df_inc, bound, self.risk)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


'''new'''
class MzFixRiskBootBlAllocate(MzBootBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, period=1, risk=None, bound=None, bootstrap_count=0):

        super(MzFixRiskBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, bound, bootstrap_count)

        self._risk = risk

    @property
    def risk(self):

        return self._risk

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)

        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        ws = self.markowitz_bootstrap_bl_fixrisk(day, df_inc, P, eta, alpha, bound, self.risk, self.bootstrap_count)

        return ws

    def markowitz_bootstrap_bl_fixrisk(self, day, df_inc, P, eta, alpha, bound, target_risk, bootstrap_count):

        look_back = len(df_inc)
        loop_num = self.get_loop_num(bootstrap_count, look_back)

        day_indices = self.create_day_indices(look_back, loop_num)

        args = (df_inc, P, eta, alpha, bound, target_risk)
        v_markowitz_random_bl_fixrisk = np.vectorize(partial(self.markowitz_random_bl_fixrisk, *args), signature='(n)->(m)')
        ws = np.mean(v_markowitz_random_bl_fixrisk(day_indices), axis=0)
        ws = pd.Series(ws, index=df_inc.columns, name=day)

        return ws

    def markowitz_random_bl_fixrisk(self, df_inc, P, eta, alpha, bound, target_risk, random_index):

        tmp_df_inc = df_inc.iloc[random_index.tolist()]
        ev_cov = self.calc_ev_cov(tmp_df_inc)
        ev_ret = self.calc_ev_ret(tmp_df_inc, ev_cov, P, eta, alpha)
        ws = self.markowitz_bl_fixrisk(ev_cov, ev_ret, bound, target_risk)

        return ws

    def markowitz_bl_fixrisk(self, ev_cov, ev_ret, bound, target_risk):

        num_assets = ev_ret.shape[0]
        w0 = np.full(num_assets, 1.0/num_assets)

        bnds = [(bnd['lower'], bnd['upper']) for bnd in bound]

        asset_sum1_limit = 0.0
        sum1_limit_assets = []
        for asset in range(len(bound)):
            if bound[asset]['sum1'] != 0.0:
                sum1_limit_assets.append(asset)
                asset_sum1_limit = bound[asset]['sum1']

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        if target_risk is not None:
            cons.append({'type': 'ineq', 'fun': lambda x: target_risk - np.sqrt(np.dot(x, np.dot(ev_cov, x)))})
        if asset_sum1_limit > 0.0:
            cons.append({'type': 'ineq', 'fun': lambda x: asset_sum1_limit - np.sum(x[sum1_limit_assets])})
        cons = tuple(cons)

        res = scipy.optimize.minimize(self.risk_budget_objective, w0, args=(ev_ret, ev_cov, target_risk), method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False, 'eps': 1e-3})

        # final_risk = np.sqrt(np.dot(res.x, np.dot(ev_cov, res.x)))
        # final_ret = np.dot(res.x, ev_ret)
        # final_ws = res.x
        # final_sharpe = (final_ret - Const.rf) / final_risk
        # final_res = np.append(final_ws, np.array([final_risk, final_ret, final_sharpe]))

        return res.x

    def calc_ev_cov(self, df_inc):

        ev_cov = df_inc.cov().values

        return ev_cov

    def calc_ev_ret(self, df_inc, ev_cov, P, eta, alpha):

        if eta.size == 0:
            ev_ret = df_inc.mean().values
        else:
            initialvalue = np.mean(df_inc.T.values, axis=1)
            ev_ret = fin.black_litterman(initialvalue, ev_cov, P, eta, alpha).reshape(-1)

        return ev_ret

    def risk_budget_objective(self, x, ev_ret, ev_cov, target_risk):

        return -np.dot(ev_ret, x)


'''old'''
class MzFixRiskBootWaveletAllocate(Allocate):

    def __init__(self, globalid, assets, wavelet_assets, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzFixRiskBootWaveletAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self._cpu_count = cpu_count
        else:
            self._cpu_count = cpu_count
        self._bootstrap_count = bootstrap_count
        self.risk = risk
        self.wavelet_assets = wavelet_assets


    def allocate_algo(self, day, df_inc, bound):
        wavelet_asset_ids = ['120000001','120000002', '120000010', '120000011' ,'120000016', '120000080', '120000081', '120000082', 'ERI000001', 'ERI000002']
        wavelet_df_inc, wavelet_bound = self.load_wavelet_allocate_data(day, wavelet_asset_ids)
        wavelet_df_inc[wavelet_df_inc > 0] = 0.0
        print(wavelet_df_inc)
        risk, returns, ws, sharpe = PF.markowitz_r_spe(wavelet_df_inc, wavelet_bound)
        #risk, returns, ws, sharpe = PF.markowitz_bootstrap_smooth(wavelet_df_inc, wavelet_bound, 26, cpu_count = self._cpu_count, bootstrap_count = self._bootstrap_count)
        wavelet_ws = dict(zip(wavelet_df_inc.columns.ravel(), ws))
        cols = []
        for asset_id in df_inc.columns:
            if asset_id in set(['120000039']):
                cols.append(asset_id)
        if len(cols) == 0:
            return wavelet_ws

        wavelet_inc = pd.Series(np.zeros(len(df_inc)), index = df_inc.index)
        for asset_id in wavelet_ws:
            pos = wavelet_ws[asset_id]
            wavelet_inc = wavelet_inc + df_inc[asset_id] * pos

        fix_risk_asset_inc = df_inc[cols]
        fix_risk_asset_inc['wavelet_inc'] = wavelet_inc
        fix_risk_asset_bound = []
        for i in range(0, len(fix_risk_asset_inc.columns)):
            asset_bound = bound[i]
            asset_bound['upper'] = 1.0
            fix_risk_asset_bound.append(asset_bound)

        risk, returns, fix_risk_asset_ws, sharpe = PF.markowitz_bootstrap_fixrisk(fix_risk_asset_inc, fix_risk_asset_bound, self.risk, cpu_count = self._cpu_count, bootstrap_count = self._bootstrap_count)
        fix_risk_asset_ws = dict(zip(fix_risk_asset_inc.columns.ravel(), fix_risk_asset_ws))
        if fix_risk_asset_ws['120000039'] < 0.25:
            fix_risk_asset_ws['wavelet_inc'] = fix_risk_asset_ws['wavelet_inc'] + fix_risk_asset_ws['120000039']
            fix_risk_asset_ws['120000039'] = 0
        final_ws = {}
        for key in fix_risk_asset_ws.keys():
            if key == 'wavelet_inc':
                for wavelet_asset_id in wavelet_ws.keys():
                    final_ws[wavelet_asset_id] = wavelet_ws[wavelet_asset_id] * fix_risk_asset_ws[key]
            else:
                final_ws[key] = fix_risk_asset_ws[key]

        return final_ws

    def allocate(self):


        adjust_days = self.index[self.lookback - 1::self.period]
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

        return pos_df


    def load_wavelet_allocate_data(self, day ,asset_ids):

        reindex = self.index[self.index <= day][-1 * self.lookback:]

        bound = []
        allocate_asset_ids = []
        for asset_id in asset_ids:
            asset_bound = AssetBound.get_asset_day_bound(asset_id, day, self.bound).to_dict()
            if asset_bound['upper'] > 0:
                bound.append(asset_bound)
                allocate_asset_ids.append(asset_id)

        data = {}
        for asset_id in allocate_asset_ids:
            data[asset_id] = self.wavelet_assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc  = df_nav.pct_change().fillna(0.0)
        #df_inc  = np.log(df_nav).diff().fillna(0.0)

        return df_inc, bound


'''new'''
class MzFixRiskBootWaveletBlAllocate(MzFixRiskBootBlAllocate):

    def __init__(self, globalid, assets, wavelet_assets, views, reindex, lookback, period=1, risk=None, bound=None, bootstrap_count=0):

        super(MzFixRiskBootWaveletBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, risk, bound, bootstrap_count)

        self._wavelet_assets = wavelet_assets

    @property
    def wavelet_assets(self):

        return self._wavelet_assets

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)

        wavelet_df_inc, wavelet_bound = self.load_wavelet_allocate_data(day, list(self.assets.keys()))
        df_inc = df_inc + wavelet_df_inc
        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        ws = self.markowitz_bootstrap_bl_fixrisk(day, df_inc, P, eta, alpha, bound, self.risk, self.bootstrap_count)

        return ws

    def load_wavelet_allocate_data(self, day, asset_ids):

        reindex = self.index[self.index <= day][-1 * self.lookback:]

        bound = []
        allocate_asset_ids = []
        for asset_id in asset_ids:
            asset_bound = AssetBound.get_asset_day_bound(asset_id, day, self.bound).to_dict()
            if asset_bound['upper'] > 0:
                bound.append(asset_bound)
                allocate_asset_ids.append(asset_id)

        data = {}
        for asset_id in allocate_asset_ids:
            data[asset_id] = self.wavelet_assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc = df_nav.pct_change().fillna(0.0)

        return df_inc, bound


class FactorValidAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, cpu_count = None):

        super(FactorValidAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, stock_ids = assets.keys(), begin_date = '2010-01-01')
        self.sfr = load_stock_factor_return(sf_ids = sf_ids, begin_date = '2012-01-01')

        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self._cpu_count = cpu_count
        else:
            self._cpu_count = cpu_count

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        pool = Pool(self._cpu_count)
        wss = pool.map(self.allocate_algo, adjust_days)
        pool.close()
        pool.join()

        for day, ws in zip(adjust_days, wss):
            for asset_id in ws.keys():
                pos_df.loc[day, asset_id] = ws[asset_id]

        return pos_df


    def allocate_algo(self, day):

        begin_date = (day.date() - timedelta(self.lookback - 1)).strftime('%Y-%m-%d')
        end_date = day.date().strftime('%Y-%m-%d')

        sfe = self.sfe[(self.sfe.index.get_level_values(2) > begin_date) & (self.sfe.index.get_level_values(2) < end_date)].reset_index()
        sfr = self.sfr[(self.sfr.index.get_level_values(1) > begin_date) & (self.sfr.index.get_level_values(1) < end_date)].reset_index()

        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)
        sfe = sfe[self.sf_ids]

        sfr = sfr.set_index(['trade_date', 'sf_id'])
        sfr = sfr.unstack()
        sfr.columns = sfr.columns.droplevel(0)
        sfr = sfr[self.sf_ids]

        R = sfr.mean()
        factor_weights = np.sign(R.values) * 0.5

        stock_weights = PureFactor.cal_weight(sfe, factor_weights)
        ws = dict(zip(sfe.index, stock_weights))

        return ws


class FactorIndexAllocate(Allocate):

    def __init__(self, globalid, reindex, lookback, assets = None, period = 1, bound = None, target = None):

        super(FactorIndexAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        if target is None:
            self.target = [1] + [0] * 8
        else:
            self.target = target


        self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, begin_date = '2007-10-01')

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]

        df = pd.DataFrame()
        pos_df = {}

        pool = Pool(8)
        wss = pool.map(self.allocate_algo, adjust_days)
        pool.close()
        pool.join()

        # wss = []
        # for day in adjust_days:
        #     wss.append(self.allocate_algo(day))

        for day, ws in zip(adjust_days, wss):
            pos_df[day] = ws

        pos_df = df.from_dict(pos_df, orient = 'index')
        pos_df = pos_df.fillna(0.0)

        return pos_df

    def allocate_algo(self, day):

#        index_pos = asset_stock.load_index_pos('2070000191', day) 此处为原caihui码
        index_pos = asset_stock.load_index_pos('000906.SH', day)
        asset_ids = self.sfe.index.levels[0].intersection(index_pos).values

        begin_date = (day.date() - timedelta(self.lookback - 1)).strftime('%Y-%m-%d')
        end_date = day.date().strftime('%Y-%m-%d')

        sfe = self.sfe[(self.sfe.index.get_level_values(2) > begin_date) & (self.sfe.index.get_level_values(2) < end_date)]
        sfe = sfe.loc[asset_ids]
        sfe = sfe.reset_index()
        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)
        print(day)
        sfe = sfe[self.sf_ids]

        stock_weights = IndexFactor.cal_weight(sfe, self.target)

        ws = dict(zip(sfe.index, stock_weights))

        return ws


'''new'''
class MzALayerFixRiskBootBlAllocate(MzFixRiskBootBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, period=1, risk=None, bound=None, bootstrap_count=0):

        super(MzALayerFixRiskBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, risk, bound, bootstrap_count)

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)

        df_inc_layer, layer_ws, bound = self.load_a_layer(day, df_inc)
        P, eta, alpha = self.load_bl_view(day, df_inc_layer.columns)
        ws = self.markowitz_bootstrap_bl_fixrisk(day, df_inc_layer, P, eta, alpha, bound, self.risk, self.bootstrap_count)
        ws = self.calc_a_layer_weight(ws, layer_ws)

        return ws

    def load_a_layer(self, day, df_inc):

        layer_assets_1 = ['120000053', '120000056','120000058','120000073']
        layer_assets_2 = ['MZ.FA0010', 'MZ.FA0050','MZ.FA0070']
        layer_assets = layer_assets_1 + layer_assets_2

        layer_assets = dict([(asset_id , Asset(asset_id)) for asset_id in layer_assets])
        layer_bounds = {}
        for asset in layer_assets.keys():
            layer_bounds[asset] = self.bound[asset]
        rp_allocate = RpAllocate('ALC.000001', layer_assets, self.index, self.lookback, bound = layer_bounds)
        
        layer_ws, df_alayer_inc = rp_allocate.allocate_day(day)

        df_inc['ALayer'] = df_alayer_inc
        df_inc_layer = df_inc[df_inc.columns.difference(layer_assets)]

        bound = []
        allocate_asset_ids = []
        for asset_id in df_inc_layer.columns:
            asset_bound = AssetBound.get_asset_day_bound(asset_id, day, self.bound).to_dict()
            if asset_bound['upper'] > 0:
                bound.append(asset_bound)
                # allocate_asset_ids.append(asset_id)

        return df_inc_layer, layer_ws, bound

    def calc_a_layer_weight(self, ws, layer_ws):

        for asset in layer_ws.index:
            ws.loc[asset] = ws.loc['ALayer'] * layer_ws.loc[asset]
        ws.drop(labels=['ALayer'], inplace=True)

        return ws


class SingleValidFactorAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, alloc_num, bound=None, period=1):
        super(SingleValidFactorAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.alloc_num = alloc_num

    def allocate_algo(self, day, df_inc, bound):

        df_mean = df_inc.mean()
        df_pos = df_mean.copy()
        valid_factors = df_mean.nlargest(self.alloc_num).index
        invalid_factors = df_mean.index.difference(valid_factors)
        df_pos.loc[valid_factors] = 1.0 / self.alloc_num
        df_pos.loc[invalid_factors] = 0.0
        ws = df_pos.to_dict()

        return ws


class MonetaryAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period=1, bound=None, alloc_num=2):
        super(MonetaryAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        mnf = MonetaryFundFilter()
        mnf.handle()
        self.mnf = mnf
        self.alloc_num = alloc_num

    def allocate_algo(self, day, df_inc, bound):

        fund_status = self.mnf.fund_status
        fund_status = fund_status[fund_status.fi_yingmi_amount <= 1e3]
        fund_status = fund_status[fund_status.fi_yingmi_subscribe_status == 0.0]
        valid_ids_1 = fund_status.index
        fund_fee = self.mnf.fund_fee.ff_fee
        valid_ids_2 = fund_fee[fund_fee >= 0.2].index
        valid_ids_2 = [str(fund_code) for fund_code in valid_ids_2]
        valid_ids = np.intersect1d(valid_ids_1, valid_ids_2)

        tmp_scale = self.mnf.fund_scale.loc[day]
        tmp_scale = tmp_scale.sort_values(ascending=False)
        scale_filter_codes = tmp_scale[tmp_scale > 1e10].index
        scale_filter_ids = [str(self.mnf.fund_id_dict[fund_code]) for fund_code in scale_filter_codes]

        final_filter_ids = np.intersect1d(scale_filter_ids, valid_ids)
        tmp_df_inc = df_inc.copy()
        tmp_df_inc = tmp_df_inc[final_filter_ids]

        ws = {}
        num = self.alloc_num
        rs = tmp_df_inc.mean()
        rs = rs.sort_values(ascending=False)
        num = min(num, len(rs))
        for i in range(0, num):
            fund_globalid = rs.index[i]
            ws[fund_globalid] = 1.0 / num
            # print(fund_globalid, fund_fee.loc[int(fund_globalid)])

        # print(ws)
        return ws

    @staticmethod
    def all_monetary_fund_globalid():
        all_monetary_fund_df = base_ra_fund.find_type_fund(3)
        all_monetary_fund_df = all_monetary_fund_df.set_index(['globalid'])
        return all_monetary_fund_df.index.astype(str).tolist()


class CppiAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period=1, bound=None, forcast_days=90, var_percent=10):
        super(CppiAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.forcast_days = forcast_days
        self.var_percent = var_percent

    def allocate_cppi(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        sdate = adjust_days[0].date()

        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(columns=asset_ids)

        # allocate monetary fund for first 3 years
        edate_3m = sdate + timedelta(32)
        edate_3m = datetime(edate_3m.year, edate_3m.month, edate_3m.day)
        pre_3m = adjust_days[adjust_days <= edate_3m]
        adjust_days = adjust_days[adjust_days > edate_3m]
        pos_df = pd.DataFrame(columns=asset_ids, index=pre_3m)
        pos_df = pos_df.fillna(0.0)
        pos_df['120000039'] = 1.0

        self.bond_view =  asset_ra_bl_view.load('BL.000002','120000010')
        self.bond_view.sort_index(inplace = True)
        self.pos_df = pos_df
        for day in adjust_days:
            print(day)

            df_inc, bound = self.load_allocate_data(day, asset_ids)
            ws = self.allocate_algo(day, df_inc, bound)
            view = 0
            bond_view = self.bond_view.loc[self.bond_view.index <= day.date()]
            if len(bond_view) > 0:
                view = bond_view.iloc[-1].ravel()[0]
            a = ws['PO.LRB010']
            b = ws['PO.IB0010']
            if a + b == 0.0:
                ws['120000039'] = 1.0
                ws['PO.LRB010'] = 0.0
                ws['PO.IB0010'] = 0.0
            else:
                if view == 1:
                    ws['120000039'] = ws['120000039'] * 0.5
                    ws['PO.LRB010'] = (1 - ws['120000039']) * a/(a+b)
                    ws['PO.IB0010'] = (1 - ws['120000039']) * b/(a+b)
                elif view == -1:
                    ws['120000039'] = ws['120000039'] * 1.5
                    ws['PO.LRB010'] = (1 - ws['120000039']) * a/(a+b)
                    ws['PO.IB0010'] = (1 - ws['120000039']) * b/(a+b)
                else:
                    pass

            for asset_id in list(ws.keys()):
                pos_df.loc[day, asset_id] = ws[asset_id]
            self.pos_df = pos_df
        return pos_df

    def allocate_algo(self, day, df_inc, bound):

        bond_var = CppiAllocate.bond_var(self.forcast_days, self.var_percent, day)
        monetary_ret = 1.04**(self.forcast_days / 365) - 1

        df_pos = self.pos_df.copy()
        df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_pos, result_col='portfolio')
        df_nav = df_nav_portfolio.portfolio

        tmp_ret = df_nav.iloc[-1] / df_nav.iloc[0]
        tmp_benchmark_ret = 0
        tmp_overret = tmp_ret - tmp_benchmark_ret

        tmp_dd = 1 - df_nav.iloc[-1] / df_nav.iloc[-30:].max()
        money_pos = df_pos.iloc[-1].loc['120000039']
        # print(tmp_dd)

        if (tmp_dd > 0.01) or (money_pos == 1.0 and tmp_dd > 0):
            ws = CppiAllocate.money_alloc()
        elif tmp_overret <= 0:
            ws = CppiAllocate.money_alloc()
        else:
            # 未来3个月有90%的概率战胜净值不跌到余额宝之下
            lr = 1 / (monetary_ret - bond_var)
            ws = CppiAllocate.cppi_alloc(tmp_overret, tmp_ret, lr)
        # print(day, ws)
        # print(day, tmp_overret, tmp_ret)
        return ws

    @staticmethod
    def money_alloc():

        ws = {}
        ws['PO.IB0010'] = 0.0
        ws['PO.LRB010'] = 0.0
        ws['120000039'] = 1.0

        return ws

    @staticmethod
    def cppi_alloc(overret, tnav, lr):

        ws = {}
        ws_b = (overret - 1 + 0.006) / tnav * lr
        ws_m = 1 - ws_b
        if ws_m > 0.20:
            ws['PO.IB0010'] = ws_b * 0.2
            ws['PO.LRB010'] = ws_b * 0.8
            ws['120000039'] = ws_m
        else:
            ws['PO.IB0010'] = 0.8 * 0.2
            ws['PO.LRB010'] = 0.8 * 0.8
            ws['120000039'] = 0.2

        return ws

    @staticmethod
    def bond_var(period, percent, day):
        
        df_nav = base_ra_index_nav.load_series('120000011')
        df_inc = df_nav.pct_change(period)
        df_inc = df_inc.dropna()
        df_inc = df_inc[df_inc.index < day]
        df_inc = df_inc.iloc[-365*5:]
        var = np.percentile(df_inc, percent)

        #ib_inc = Asset('PO.IB0010').nav().pct_change()
        #lrb_inc = Asset('PO.LRB010').nav().pct_change()
        #dates = ib_inc.index & lrb_inc.index
        #ib_inc = ib_inc.loc[dates]
        #lrb_inc = lrb_inc.loc[dates]
        #ib_lrb_inc = ib_inc * 0.2 + lrb_inc * 0.8
        #ib_lrb_inc = ib_lrb_inc.reindex(df_inc.index).fillna(0.0)
        #ib_lrb_inc[ib_lrb_inc == 0] = df_inc[ib_lrb_inc == 0]
        #df_inc = ib_lrb_inc
        #df_nav = (df_inc + 1).cumprod()
        #df_inc = df_nav.pct_change(period).fillna(0.0)
        #df_inc = df_inc[df_inc.index < day]
        #df_inc = df_inc.iloc[-365*5:]
        #var = np.percentile(df_inc, percent)

        return var


'''new'''
class MzLayerFixRiskBootBlAllocate(MzFixRiskBootBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, period=1, data_period='week', risk=None, bound=None, bootstrap_count=0):

        super(MzLayerFixRiskBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, risk, bound, bootstrap_count)

        if data_period == 'week':
            pass
        elif data_period == 'day':
            self._risk /= np.sqrt(5)
            self.load_allocate_data = partial(self.load_allocate_data, data_period=data_period)
        else:
            raise ValueError

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)

        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        ws = self.markowitz_bootstrap_bl_fixrisk(day, df_inc, P, eta, alpha, bound, self.risk, self.bootstrap_count)
        # ws = self.calc_layer_weight(day, ws)

        return ws

    def calc_layer_weight(self, day, ws):

        for asset_id in ws.keys():
            if asset_id[0:2] == 'MZ' and asset_id[3:5] != 'FA':
                layer_ws = self.df_mz_pos.loc[asset_id, day]
                for sub_asset_id in sub_assets:
                    ws.loc[sub_asset_id] = ws.loc[asset_id] * layer_ws.loc[sub_asset_id]
                ws.drop(labels=[asset_id], inplace=True)

        return ws


'''new'''
class MzLayerFixRiskSmoothBootBlAllocate(MzLayerFixRiskBootBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, smooth=0, period=1, data_period='week', risk=None, bound=None, bootstrap_count=0):

        super(MzLayerFixRiskSmoothBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, data_period, risk, bound, bootstrap_count)

        self._smooth = smooth

    @property
    def smooth(self):

        return self._smooth

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        _, df_inc, bound = self.load_allocate_data(day, asset_ids)

        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        ws = self.markowitz_smooth_bootstrap_bl_fixrisk(day, df_inc, P, eta, alpha, bound, self.risk, self.smooth, self.bootstrap_count)
        # ws = self.calc_layer_weight(day, ws)

        return ws

    def markowitz_smooth_bootstrap_bl_fixrisk(self, day, df_inc, P, eta, alpha, bound, target_risk, smooth, bootstrap_count):

        look_back = len(df_inc)
        smooth, look_back_smooth = self.calc_look_back_smooth(smooth, look_back)
        loop_num = self.get_loop_num(bootstrap_count, look_back_smooth)

        day_indices = self.create_smooth_day_indices(look_back, look_back_smooth, smooth, loop_num)

        args = (df_inc, P, eta, alpha, bound, target_risk)
        v_markowitz_random_bl_fixrisk = np.vectorize(partial(self.markowitz_random_bl_fixrisk, *args), signature='(n)->(m)')
        ws = np.mean(v_markowitz_random_bl_fixrisk(day_indices), axis=0)
        ws = pd.Series(ws, index=df_inc.columns, name=day)

        return ws

    def calc_look_back_smooth(self, smooth, look_back):

        if smooth % 2:
            smooth += 1
        smooth = min(smooth, look_back)
        look_back_smooth = look_back - smooth // 2

        return smooth, look_back_smooth

    def create_smooth_day_indices(self, look_back, look_back_smooth, smooth, loop_num):

        rep_num = loop_num * (look_back_smooth // 2) // look_back_smooth

        day_indices = list(range(smooth, look_back)) * rep_num
        for i in range(smooth):
            day_index = [i] * round((i+1)/(smooth+1)*rep_num)
            day_indices.extend(day_index)
        day_indices.sort()
        random.shuffle(day_indices)

        day_indices = np.array(day_indices)
        day_indices = day_indices.reshape(len(day_indices) // (look_back_smooth // 2), look_back_smooth // 2)

        return day_indices


'''new'''
class MzLayerFixRiskCovSmoothBootBlAllocate(MzLayerFixRiskSmoothBootBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, smooth=0, period=1, data_period='week', risk=None, cov_algo=None, bound=None, bootstrap_count=0):

        super(MzLayerFixRiskCovSmoothBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, smooth, period, data_period, risk, bound, bootstrap_count)

        if cov_algo is not None:
            if not hasattr(self, f'calc_ev_cov_{cov_algo}'):
                raise ValueError
            self.calc_ev_cov = getattr(self, f'calc_ev_cov_{cov_algo}')

    def calc_ev_cov_empirical(self, df_inc):

        ev_cov = empirical_covariance(df_inc)

        return ev_cov

    def calc_ev_cov_ledoit_wolf(self, df_inc):

        ev_cov = ledoit_wolf(df_inc, assume_centered=False)[0]

        return ev_cov


'''new'''
class MzLayerFixRiskCovSampleBlAllocate(MzLayerFixRiskCovSmoothBootBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, smooth=0, period=1, data_period='week', risk=None, cov_algo=None, bound=None, bootstrap_count=0):

        super(MzLayerFixRiskCovSampleBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, smooth, period, data_period, risk, cov_algo, bound, bootstrap_count)

    def allocate_algo(self, day):

        asset_ids = list(self.assets.keys())
        df_nav, df_inc, bound = self.load_allocate_data(day, asset_ids)
        ev_cov = self.calc_ev_cov(df_inc)
        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        ws = self.markowitz_sample_bl_fixrisk(day, df_nav, ev_cov, P, eta, alpha, bound, self.risk)

        return ws

    def markowitz_sample_bl_fixrisk(self, day, df_nav, ev_cov, P, eta, alpha, bound, risk):

        resample_num = 1000
        series_num = df_nav.shape[0]

        copula = GaussianCopula.gaussian_copula(df_nav, series_num, series_num*resample_num, 0)
        copula = copula.reshape(resample_num, series_num, -1)

        vvv = np.vectorize(partial(self.markowitz_resample_bl_fixrisk, ev_cov, P, eta, alpha, bound, risk), signature='(n,m)->(m)')
        ws = vvv(copula)

        # ws = []
        # for i in range(resample_num):
            # df_inc_resample = pd.DataFrame(copula[i*series_num:(i+1)*series_num], index=df_nav.index, columns=df_nav.columns)
            # ws.append(self.markowitz_resample_bl_fixrisk(ev_cov, P, eta, alpha, bound, risk, df_inc_resample))

        ws = np.array(ws)
        ws = np.mean(ws, axis=0)
        ws = pd.Series(ws, index=df_nav.columns, name=day)

        return ws

    def markowitz_resample_bl_fixrisk(self, ev_cov, P, eta, alpha, bound, target_risk, arr_inc_resample):

        df_inc_resample = pd.DataFrame(arr_inc_resample)
        ev_ret = self.calc_ev_ret(df_inc_resample, ev_cov, P, eta, alpha)
        ws = self.markowitz_bl_fixrisk(ev_cov, ev_ret, bound, target_risk)

        return ws


if __name__ == '__main__':

    print(MonetaryAllocate.all_monetary_fund_globalid())

    asset = Asset('120000001')

    asset = WaveletAsset('120000013', 2)

    trade_date = ATradeDate.week_trade_date(begin_date = '2012-01-01')

    asset_globalids = ['120000001', '120000002', '120000013', '120000014', '120000015']
    assets = {}
    for asset_id in asset_globalids:
        assets[asset_id] = Asset(asset_id)
        #assets[asset_id] = WaveletAsset(asset_id, 2)

    allocate = AvgAllocate('ALC.000001', assets, trade_date, 14)
    #print allocate.allocate().tail()

    allocate = MzAllocate('ALC.000002', assets, trade_date, 14)
    #print allocate.allocate().tail()

    allocate = MzBootAllocate('ALC.000002', assets, trade_date, 14)
    #print allocate.allocate().tail()


    view_df = View.load_view('BL.000001')

    asset_globalids = ['120000001', '120000002', 'ERI000001', '120000014', 'ERI000002']
    assets = {}
    for asset_id in asset_globalids:
        assets[asset_id] = Asset(asset_id)

    views = {}
    for asset_id in asset_globalids:
        views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = 0.5) if asset_id in view_df.columns else View(None, asset_id, confidence = 0.5)

    allocate = MzBootBlAllocate('ALC.000002', assets, views, trade_date, 26)

