#coding=utf8


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
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl
from util import xdict
from util.xdebug import dd

from asset import Asset, WaveletAsset
from allocate import Allocate, AssetBound
from asset_allocate import AvgAllocate, MzAllocate, MzBootAllocate, MzBootBlAllocate, MzBlAllocate, MzBootDownRiskAllocate, FactorValidAllocate, MzFixRiskBootAllocate, MzFixRiskBootBlAllocate, MzFixRiskBootWaveletAllocate, MzFixRiskBootWaveletBlAllocate, FactorIndexAllocate, MzALayerFixRiskBootBlAllocate, SingleValidFactorAllocate, MonetaryAllocate, CppiAllocate
from asset_allocate import *
from trade_date import ATradeDate
from view import View


import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.option('--full/--no-full', 'optfull', default=False, help=u'include all instance')
@click.option('--new/--no-new', 'optnew', default=False, help=u'use new framework')
@click.option('--append/--no-append', 'optappend', default=False, help=u'append pos or not')
@click.option('--id', 'optid', help=u'specify markowitz id')
@click.option('--name', 'optname', default=None, help=u'specify markowitz name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.option('--start-date', 'startdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--lookback', type=int, default=26, help=u'howmany weeks to lookback')
@click.option('--adjust-period', type=int, default=1, help=u'adjust every how many weeks')
@click.option('--turnover', 'optturnover', type=float, default=0, help=u'fitler by turnover')
@click.option('--bootstrap/--no-bootstrap', 'optbootstrap', default=True, help=u'use bootstrap or not')
@click.option('--bootstrap-count', 'optbootcount', type=int, default=0, help=u'use bootstrap or not')
@click.option('--cpu-count', 'optcpu', type=int, default=0, help=u'how many cpu to use, (0 for all available)')
@click.option('--wavelet/--no-wavelet', 'optwavelet', default=False, help=u'use wavelet filter or not')
@click.option('--wavelet-filter-num', 'optwaveletfilternum', default=2, help=u'use wavelet filter num')
@click.option('--short-cut', type=click.Choice(['high', 'low', 'default']))
@click.option('--assets', multiple=True, help=u'assets')
@click.pass_context
def markowitz(ctx, optnew, optappend, optfull, optid, optname, opttype, optreplace, startdate, enddate, lookback, adjust_period, optturnover, optbootstrap, optbootcount, optwavelet, optwaveletfilternum, optcpu, short_cut, assets):

    '''markowitz group
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        if optnew:
            ctx.invoke(pos, optid=optid, optappend=optappend, sdate = startdate, edate = enddate)
            ctx.invoke(nav, optid=optid)
            ctx.invoke(turnover, optid=optid)
        else:
            if optfull is False:
                ctx.invoke(allocate, optid=optid, optname=optname, opttype=opttype, optreplace=optreplace, startdate=startdate, enddate=enddate, lookback=lookback, adjust_period=adjust_period, turnover=optturnover, optbootstrap=optbootstrap, optbootcount=optbootcount, optcpu=optcpu, optwavelet = optwavelet, optwaveletfilternum = optwaveletfilternum, short_cut=short_cut, assets=assets)
                ctx.invoke(nav, optid=optid)
                ctx.invoke(turnover, optid=optid)
            else:
                ctx.invoke(nav, optid=optid)
                ctx.invoke(turnover, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass


@markowitz.command(name='import')
@click.option('--id', 'optid', type=int, help=u'specify markowitz id')
@click.option('--name', 'optname', help=u'specify markowitz name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.argument('csv', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=False), required=True)
@click.pass_context
def import_command(ctx, csv, optid, optname, opttype, optreplace):
    '''
    import fund pool from csv file
    '''

    #
    # 处理id参数
    #
    if optid is not None:
        #
        # 检查id是否存在
        #
        df_existed = asset_mz_markowitz.load([str(optid)])
        if not df_existed.empty:
            s = 'markowitz instance [%s] existed' % str(optid)
            if optreplace:
                click.echo(click.style("%s, will replace!" % s, fg="yellow"))
            else:
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
            return -1;
    else:
        #
        # 自动生成id
        #
        today = datetime.now()
        prefix = '50' + today.strftime("%m%d");
        if opttype == '9':
            between_min, between_max = ('%s90' % (prefix), '%s99' % (prefix))
        else:
            between_min, between_max = ('%s00' % (prefix), '%s89' % (prefix))

        max_id = asset_mz_markowitz.max_id_between(between_min, between_max)
        if max_id is None:
            optid = between_min
        else:
            if max_id >= between_max:
                s = "run out of instance id [%d]" % max_id
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
                return -1

            if optreplace:
                optid = max_id
            else:
                optid = max_id + 1;

    #
    # 处理name参数
    #
    if optname is None:
        optname = os.path.basename(csv);


    db = database.connection('asset')
    metadata = MetaData(bind=db)
    mz_markowitz = Table('mz_markowitz', metadata, autoload=True)
    mz_markowitz_pos = Table('mz_markowitz_pos', metadata, autoload=True)
    mz_markowitz_nav = Table('mz_markowitz_nav', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        mz_markowitz.delete(mz_markowitz.c.globalid == optid).execute()
        mz_markowitz_pos.delete(mz_markowitz_pos.c.mz_markowitz_id == optid).execute()
        mz_markowitz_nav.delete(mz_markowitz_nav.c.mz_markowitz_id == optid).execute()

    now = datetime.now()
    #
    # 导入数据
    #
    row = {
        'globalid': optid, 'mz_type':opttype, 'mz_name': optname,
        'mz_pool': '', 'mz_reshape': '', 'created_at': func.now(), 'updated_at': func.now()
    }
    mz_markowitz.insert(row).execute()

    df = pd.read_csv(csv, parse_dates=['date'])
    df['risk'] = (df['risk'] * 10).astype(int)
    renames = dict(
        {'date':'mz_date', 'risk':'mz_alloc_id'}.items() + DFUtil.categories_types(as_int=True).items()
    )
    df = df.rename(columns=renames)
    df['mz_markowitz_id'] = optid

    df.set_index(['mz_markowitz_id', 'mz_alloc_id', 'mz_date'], inplace=True)

    # 四舍五入到万分位
    df = df.round(4)
    # 过滤掉过小的份额
    df[df.abs() < 0.0009999] = 0
    # 补足缺失
    df = df.apply(npu.np_pad_to, raw=True, axis=1)
    # 过滤掉相同
    df = df.groupby(level=(0,1), group_keys=False).apply(DFUtil.filter_same_with_last)

    df.columns.name='mz_asset'
    df_tosave = df.stack().to_frame('mz_ratio')
    df_tosave = df_tosave.loc[df_tosave['mz_ratio'] > 0, ['mz_ratio']]
    if not df_tosave.empty:
        database.number_format(df_tosave, columns=['mz_ratio'], precision=4)

    df_tosave['updated_at'] = df_tosave['created_at'] = now

    df_tosave.to_sql(mz_markowitz_pos.name, db, index=True, if_exists='append', chunksize=500)

    if len(df_tosave.index) > 1:
        logger.info("insert %s (%5d) : %s " % (mz_markowitz_pos.name, len(df_tosave.index), df_tosave.index[0]))

    click.echo(click.style("import complement! instance id [%s]" % (optid), fg='green'))

    return 0

@markowitz.command()
@click.option('--id', 'optid', type=int, help=u'specify markowitz id')
@click.option('--name', 'optname', default=None, help=u'specify markowitz name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.option('--start-date', 'startdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--lookback', type=int, default=26, help=u'howmany weeks to lookback')
@click.option('--adjust-period', type=int, default=1, help=u'adjust every how many weeks')
@click.option('--turnover', type=float, default=0, help=u'fitler by turnover')
@click.option('--bootstrap/--no-bootstrap', 'optbootstrap', default=True, help=u'use bootstrap or not')
@click.option('--bootstrap-count', 'optbootcount', type=int, default=0, help=u'use bootstrap or not')
@click.option('--cpu-count', 'optcpu', type=int, default=0, help=u'how many cpu to use, (0 for all available)')
@click.option('--wavelet/--no-wavelet', 'optwavelet', default=False, help=u'use wavelet filter or not')
@click.option('--wavelet-filter-num', 'optwaveletfilternum', default=2, help=u'use wavelet filter num')
@click.option('--short-cut', type=click.Choice(['default', 'high', 'low']))
@click.option('--algo', 'optalgo', type=click.Choice(['markowitz', 'average']), help=u'which algorithm to use for allocate')
@click.argument('assets', nargs=-1)
@click.pass_context
def allocate(ctx, optid, optname, opttype, optreplace, startdate, enddate, lookback, adjust_period, turnover,  optbootstrap, optbootcount, optcpu, optwavelet, optwaveletfilternum, short_cut, optalgo, assets):
    '''calc high low model markowitz
    '''

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1));
        enddate = yesterday.strftime("%Y-%m-%d")

    #
    # 处理id参数
    #
    if optid is not None:
        #
        # 检查id是否存在
        #
        df_existed = asset_mz_markowitz.load([str(optid)])
        if not df_existed.empty:
            s = 'markowitz instance [%s] existed' % str(optid)
            if optreplace:
                click.echo(click.style("%s, will replace!" % s, fg="yellow"))
            else:
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
                return -1;
    else:
        #
        # 自动生成id
        #
        today = datetime.now()
        prefix = '50' + today.strftime("%m%d");
        if opttype == '9':
            between_min, between_max = ('%s90' % (prefix), '%s99' % (prefix))
        else:
            between_min, between_max = ('%s00' % (prefix), '%s89' % (prefix))

        max_id = asset_mz_markowitz.max_id_between(between_min, between_max)
        if max_id is None:
            optid = between_min
        else:
            if max_id >= between_max:
                s = "run out of instance id [%s]" % max_id
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
                return -1

            if optreplace:
                optid = max_id
            else:
                optid = str(string.atoi(max_id) + 1);
    #
    # 处理assets参数
    #
    if assets:
        assets = {k: v for k,v in [parse_asset(a) for a in assets]}
    else:
        if short_cut == 'high':
            assets = {
                '120000001':  {'sum1': 0,    'sum2' : 0,   'upper': 0.70,  'lower': 0.0}, #沪深300指数修型
                '120000002':  {'sum1': 0,    'sum2' : 0,   'upper': 0.70,  'lower': 0.0}, #中证500指数修型
                #'120000013':  {'sum1': 0.65, 'sum2' : 0,   'upper': 0.35, 'lower': 0.0}, #标普500指数
                #'120000015':  {'sum1': 0.65, 'sum2' : 0,   'upper': 0.35, 'lower': 0.0}, #恒生指数修型
                '120000014':  {'sum1': 0.0, 'sum2' : 0.0,'upper': 0.7, 'lower': 0.0}, #黄金指数修型
                'ERI000002':  {'sum1': 0.0, 'sum2' : 0.0,'upper': 0.7, 'lower': 0.0}, #人民币计价恒生指数
                'ERI000001':  {'sum1': 0.0, 'sum2' : 0.0,'upper': 0.7, 'lower': 0.0}, #人民币计价标普500指数
                # 120000029:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #南华商品指数
                # 120000028:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #标普高盛原油商品指数收益率
                # 120000031:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #房地产指数
            }
            if optname is None:
                optname = u'马克维茨%s(高风险)' % today.strftime("%m%d")
        elif short_cut == 'low':
            assets = {
                '120000010':  {'sum1': 0, 'sum2': 0, 'upper': 1.0, 'lower': 0.0},
                '120000011':  {'sum1': 0, 'sum2': 0, 'upper': 1.0, 'lower': 0.0},
            }
            if optname is None:
                optname = u'马克维茨%s(低风险)' % today.strftime("%m%d")
            if optalgo is None:
                optalgo = 'average'

        else: # short_cut == 'default'
            assets = {
                '120000001':  {'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0}, #沪深300指数修型
                '120000002':  {'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0}, #中证500指数修型

                # 120000013:  {'sum1': 0.65, 'sum2' : 0,   'upper': 0.35, 'lower': 0.0}, #标普500指数
                # 120000015:  {'sum1': 0.65, 'sum2' : 0,   'upper': 0.35, 'lower': 0.0}, #恒生指数修型
                # 120000014:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.35, 'lower': 0.0}, #黄金指数修型
                # 120000029:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #南华商品指数
                # 120000028:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #标普高盛原油商品指数收益

                '120000013':  {'sum1': 0, 'sum2' : 0,   'upper': 0.55, 'lower': 0.0}, #标普500指数
                '120000015':  {'sum1': 0, 'sum2' : 0,   'upper': 0.55, 'lower': 0.0}, #恒生指数修型
                '120000014':  {'sum1': 0, 'sum2' : 0,'upper': 0.32, 'lower': 0.0}, #黄金指数修型
                '120000029':  {'sum1': 0, 'sum2' : 0,'upper': 0.20, 'lower': 0.0}, #南华商品指数
                '120000028':  {'sum1': 0, 'sum2' : 0,'upper': 0.15, 'lower': 0.0}, #标普高盛原油商品指数收益率
                '120000031':  {'sum1': 0, 'sum2' : 0,'upper': 0.20, 'lower': 0.0}, #房地产指数

            }

    today = datetime.now()
    if optname is None:
        optname = u'马克维茨%s(实验)' % today.strftime("%m%d")

    bootstrap = optbootcount if optbootstrap else None

    if optalgo == 'average':
        algo, risk = 1, 0.1
        df = average_days(startdate, enddate, assets)
    else:
        algo, risk = 3, 1.0
        df = markowitz_days(
            startdate, enddate, assets,
            label='markowitz', lookback=lookback, adjust_period=adjust_period, bootstrap=bootstrap, cpu_count=optcpu, wavelet = optwavelet, wavelet_filter_num = optwaveletfilternum)

    df_sharpe = df[['return', 'risk', 'sharpe']].copy()
    df.drop(['return', 'risk', 'sharpe'], axis=1, inplace=True)
    # print df.head()
    # print df_sharpe.head()

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    mz_markowitz        = Table('mz_markowitz', metadata, autoload=True)
    mz_markowitz_alloc  = Table('mz_markowitz_alloc', metadata, autoload=True)
    mz_markowitz_asset  = Table('mz_markowitz_asset', metadata, autoload=True)
    mz_markowitz_pos    = Table('mz_markowitz_pos', metadata, autoload=True)
    mz_markowitz_nav    = Table('mz_markowitz_nav', metadata, autoload=True)
    #mz_markowitz_sharpe = Table('mz_markowitz_sharpe', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        mz_markowitz.delete(mz_markowitz.c.globalid == optid).execute()
        mz_markowitz_asset.delete(mz_markowitz_asset.c.mz_markowitz_id == optid).execute()
        mz_markowitz_pos.delete(mz_markowitz_pos.c.mz_markowitz_id == optid).execute()
        mz_markowitz_nav.delete(mz_markowitz_nav.c.mz_markowitz_id == optid).execute()
        #mz_markowitz_sharpe.delete(mz_markowitz_sharpe.c.mz_markowitz_id == optid).execute()

    now = datetime.now()
    # 导入数据: markowitz
    row = {
        'globalid': optid, 'mz_type':opttype, 'mz_name': optname,
        'created_at': func.now(), 'updated_at': func.now()
    }
    mz_markowitz.insert(row).execute()
    # 导入数据: markowitz_alloc
    row = {
        'globalid': optid, 'mz_markowitz_id': optid, 'mz_type':opttype,
        'mz_name': optname, 'mz_algo': algo, 'mz_risk': risk,
        'created_at': func.now(), 'updated_at': func.now()
    }
    mz_markowitz_alloc.insert(row).execute()

    #
    # 导入数据: markowitz_asset
    #
    assets = {k: merge_asset_name_and_type(k, v) for (k, v) in assets.iteritems()}
    df_asset = pd.DataFrame(assets).T
    df_asset.index.name = 'mz_markowitz_asset_id'
    df_asset['mz_markowitz_id'] = optid
    df_asset.rename(inplace=True, columns={
        'upper':  'mz_upper_limit', 'lower':'mz_lower_limit',
        'sum1':'mz_sum1_limit',  'sum2': 'mz_sum2_limit'
    })
    df_asset = df_asset.reset_index().set_index(['mz_markowitz_id', 'mz_markowitz_asset_id'])
    asset_mz_markowitz_asset.save([optid], df_asset)

    #
    # 导入数据: markowitz_pos
    #
    df = df.round(4)             # 四舍五入到万分位

    #每四周做平滑
    df = df.rolling(window = 4, min_periods = 1).mean()

    df[df.abs() < 0.0009999] = 0 # 过滤掉过小的份额
    df = df.apply(npu.np_pad_to, raw=True, axis=1) # 补足缺失
    df = DFUtil.filter_same_with_last(df)          # 过滤掉相同
    if turnover >= 0.01:
        df = DFUtil.filter_by_turnover(df, turnover)   # 基于换手率进行规律

    df.index.name = 'mz_date'
    df.columns.name='mz_markowitz_asset'

    # 计算原始资产仓位
    raw_ratios = {}
    raw_assets = {}
    for asset_id in df.columns:
        raw_assets[asset_id] = asset_id
        raw_ratios[asset_id] = df[asset_id]

    df_raw_ratio = pd.DataFrame(raw_ratios, columns=df.columns)
    df_raw_asset = pd.DataFrame(raw_assets, index=df.index, columns=df.columns)

    df_tosave = pd.concat({'mz_markowitz_ratio': df, 'mz_asset_id':df_raw_asset, 'mz_ratio': df_raw_ratio}, axis=1)

    # index
    df_tosave['mz_markowitz_id'] = optid
    df_tosave = df_tosave.reset_index().set_index(['mz_markowitz_id', 'mz_date'])

    # unstack
    df_tosave = df_tosave.stack()
    df_tosave = df_tosave.loc[(df_tosave['mz_ratio'] > 0) | (df_tosave['mz_markowitz_ratio'] > 0)]

    # save
    asset_mz_markowitz_pos.save(optid, df_tosave)

    # 导入数据: markowitz_criteria
    criterias = {'return': '0001', 'risk':'0002', 'sharpe':'0003'}
    df_sharpe.index.name = 'mz_date'
    for column in df_sharpe.columns:
        criteria_id = criterias[column]
        df_criteria = df_sharpe[column].to_frame('mz_value')
        df_criteria['mz_markowitz_id'] = optid
        df_criteria['mz_criteria_id'] = criteria_id
        df_criteria = df_criteria.reset_index().set_index(['mz_markowitz_id', 'mz_criteria_id', 'mz_date'])
        asset_mz_markowitz_criteria.save(optid, criteria_id,  df_criteria)

    click.echo(click.style("markowitz allocation complement! instance id [%s]" % (optid), fg='green'))

    #
    # 在ctx中记录markowitz id 以便命令链的后面使用
    #
    ctx.obj['markowitz'] = optid
    if short_cut == 'high' or short_cut == 'default':
        ctx.obj['markowitz.high'] = optid
    if short_cut == 'low':
        ctx.obj['markowitz.low'] = optid

    return 0

def parse_asset(asset):
    segments = [s.strip() for s in asset.strip().split(':')]

    if len(segments) == 1:
        result = (int(segments[0]), {
            'upper': 1.0, 'lower': 0.0, 'sum1': 0, 'sum2': 0})
    elif len(segments) == 2:
        result = (int(segments[0]), {
            'upper': float(segments[1]), 'lower': 0.0, 'sum1': 0, 'sum2': 0})
    elif len(segments) == 3:
        result = (int(segments[0]), {
            'upper': float(segments[1]), 'lower': float(segments[2]), 'sum1': 0, 'sum2': 0})
    elif len(segments) == 4:
        result = (int(segments[0]), {
            'upper': float(segments[1]), 'lower': float(segments[2]), 'sum1': float(segments[3]), 'sum2': 0})
    else:
        if len(segments) >= 5:
            result = (int(segments[0]), {
                'upper': float(segments[1]), 'lower': float(segments[2]), 'sum1': float(segments[3]), 'sum2': float(segments[4]) })
        else:
            result = (None, {'upper': 1.0, 'lower': 0.0, 'sum1': 0, 'sum2': 0})

    return result

def merge_asset_name_and_type(asset_id, asset_data):

    if asset_id.isdigit():
        xtype = int(asset_id) // 10000000
    else:
        xtype = re.sub(r'([\d]+)','',asset_id).strip()

    if xtype == 4:
        #
        # 修型资产
        #
        asset = asset_rs_reshape.find(asset_id)
        (name, category, raw_asset) = (asset['rs_name'], asset['rs_asset'], asset['rs_asset_id'])
        (raw_name, ph) = database.load_asset_name_and_type(raw_asset)
    else:
        (name, category) = database.load_asset_name_and_type(asset_id)
        (raw_asset, raw_name) = (asset_id, name)

    return xdict.merge(asset_data, {
        'mz_asset_id': raw_asset,
        'mz_asset_name': raw_name,
        'mz_markowitz_asset_name': name,
        'mz_asset_type': category,
    })

def average_days(start_date, end_date, assets):
    '''perform markowitz asset for days
    '''

    if len(assets) > 0:
        ratio = 1.0 / len(assets)
    else:
        ratio = 0

    data = {k: {start_date: 0} for k in ['return', 'risk', 'sharpe']}

    data.update({k: {start_date: ratio} for k in assets.keys()})

    df = pd.DataFrame(data)

    df.index = pd.to_datetime(df.index)

    return df


def markowitz_days(start_date, end_date, assets, label, lookback, adjust_period, bootstrap, cpu_count=0, blacklitterman = False, wavelet = False, wavelet_filter_num = 0, markowitz_id = None):
    '''perform markowitz asset for days
    '''
    # 加载时间轴数据
    index = DBData.trade_date_index(start_date, end_date=end_date)

    # 根据调整间隔抽取调仓点
    if adjust_period:
        adjust_index = index[::adjust_period]
        if index.max() not in adjust_index:
            adjust_index = adjust_index.insert(len(adjust_index), index.max())
    else:
        adjust_index = index

    #
    # 马科维兹资产配置
    #
    s = 'perform %-12s' % label
    data = {}
    if bootstrap is None:
        count = multiprocessing.cpu_count() // 2
        process_adjust_indexs = [[] for i in range(0, count)]
        for i in range(0, len(adjust_index)):
           process_adjust_indexs[i % count].append(adjust_index[i])


        engine = database.connection('base')
        engine.dispose()
        engine = database.connection('asset')
        engine.dispose()

        manager = Manager()
        q = manager.Queue()
        processes = []
        for indexs in process_adjust_indexs:
            p = multiprocessing.Process(target = m_markowitz_day, args = (q, indexs, lookback, assets, bootstrap, cpu_count, blacklitterman ,wavelet, wavelet_filter_num, markowitz_id))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for m in range(0, q.qsize()):
            record = q.get(m)
            data[record[0]] = record[1]
    else:
        with click.progressbar(
                adjust_index, label=s.ljust(30),
                item_show_func=lambda x:  x.strftime("%Y-%m-%d") if x else None) as bar:
            for day in bar:
                # bar.update(1)
                logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))
                # 高风险资产配置
                data[day] = markowitz_day(day, lookback, assets, bootstrap, cpu_count, blacklitterman ,wavelet, wavelet_filter_num, markowitz_id)

    return pd.DataFrame(data).T


def m_markowitz_day(queue, days, lookback, assets, bootstrap, cpu_count, blacklitterman ,wavelet, wavelet_filter_num, markowitz_id):
    for day in days:
        sr = markowitz_day(day, lookback, assets, bootstrap,  cpu_count, blacklitterman ,wavelet, wavelet_filter_num, markowitz_id)
        queue.put((day, sr))


def markowitz_day(day, lookback, assets, bootstrap,  cpu_count, blacklitterman, wavelet, wavelet_filter_num, markowitz_id):
    '''perform markowitz for single day
    '''

    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)
    begin_date = index.min().strftime("%Y-%m-%d")
    end_date = index.max().strftime("%Y-%m-%d")

    #print wavelet, wavelet_filter_num
    #
    # 加载数据
    #
    data = {}
    for asset in assets:
        if wavelet:
            data[asset] = load_wavelet_nav_series(asset, index, begin_date, end_date, wavelet, wavelet_filter_num)
        else:
            data[asset] = load_nav_series(asset, index, begin_date, end_date)
    df_nav = pd.DataFrame(data).fillna(method='pad')
    df_inc  = df_nav.pct_change().fillna(0.0)

    return markowitz_r(df_inc, day, assets, bootstrap, cpu_count, blacklitterman, markowitz_id)


def markowitz_r(df_inc, day, limits, bootstrap, cpu_count, blacklitterman, markowitz_id):
    '''perform markowitz
    '''
    bound = []
    for asset in df_inc.columns:
        bound.append(limits[asset])


    if blacklitterman:
        P, eta, alpha = load_bl_view(day, df_inc, markowitz_id)

    if len(df_inc.columns) == 1:
        risk, returns, ws, sharpe = 0.0, 0.0, [1.0], 0.0
    elif (bootstrap is None) and (not blacklitterman):
        risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
    elif (bootstrap is None) and blacklitterman:
        risk, returns, ws, sharpe = PF.markowitz_r_spe_bl(df_inc, P, eta, alpha, bound)
    elif (not bootstrap is None) and (not blacklitterman):
        risk, returns, ws, sharpe = PF.markowitz_bootstrap(df_inc, bound, cpu_count=cpu_count, bootstrap_count=bootstrap)
        print(ws)
        exit(0)
    elif (not bootstrap is None) and blacklitterman:
        risk, returns, ws, sharpe = PF.markowitz_bootstrap_bl(df_inc, P, eta, alpha, bound, cpu_count=cpu_count, bootstrap_count=bootstrap)

    sr_result = pd.concat([
        pd.Series(ws, index=df_inc.columns),
        pd.Series((sharpe, risk, returns), index=['sharpe','risk', 'return'])
    ])

    return sr_result


def load_bl_view(day, df_inc, markowitz_id):

        df_argv = asset_mz_markowitz_argv.load([markowitz_id])
        df_argv.reset_index(level=0, inplace=True)
        argv = df_argv['mz_value'].to_dict()

        bl_view_id = argv['bl_view_id']

        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()

        sql = session.query(asset_ra_bl.ra_bl_view.bl_date, asset_ra_bl.ra_bl_view.bl_index_id, asset_ra_bl.ra_bl_view.bl_view).filter(asset_ra_bl.ra_bl_view.globalid == bl_view_id).filter(asset_ra_bl.ra_bl_view.bl_date <= day).statement

        view_df = pd.read_sql(sql, session.bind, index_col = ['bl_date', 'bl_index_id'], parse_dates =  ['bl_date'])
        view_df = view_df.unstack()
        view_df.columns = view_df.columns.droplevel(0)
        view_df = view_df.sort_index()

        if len(view_df) == 0:
            last_view = pd.Series(0, index = df_inc.columns)
        else:
            last_view = view_df.iloc[-1]

        session.commit()
        session.close()


        alpha = float(argv['bl_confidence'])
        views = last_view.reindex(df_inc.columns).fillna(0)
        eta = np.array(abs(views[views!=0]))
        P = np.diag(np.sign(views))
        P = np.array([i for i in P if i.sum()!=0])

        if eta.size == 0:           #If there is no view, run as non-blacklitterman
            P = alpha = None
            eta = np.array([])

        return P, eta, alpha


def load_wavelet_nav_series(asset_id, reindex=None, begin_date=None, end_date=None, wavelet=None, wavelet_filter_num=None):

    if asset_id.isdigit():
        xtype = int(asset_id) // 10000000
    else:
        xtype = re.sub(r'([\d]+)','',asset_id).strip()


    if xtype == 1:
        #
        # 基金池资产
        #
        asset_id %= 10000000
        (pool_id, category) = (asset_id // 100, asset_id % 100)
        ttype = pool_id // 10000
        sr = asset_ra_pool_nav.load_series(
            pool_id, category, ttype, reindex=None, end_date=end_date)
    elif xtype == 3:
        #
        # 基金池资产
        #
        sr = base_ra_fund_nav.load_series(
            asset_id, reindex=None, end_date=end_date)
    elif xtype == 4:
        #
        # 修型资产
        #
        sr = asset_rs_reshape_nav.load_series(
            asset_id, reindex=None, end_date=end_date)
    elif xtype == 12:
        #
        # 指数资产
        #
        sr = base_ra_index_nav.load_series(
            asset_id, reindex=None, end_date=end_date)
    elif xtype == 'ERI':

        sr = base_exchange_rate_index_nav.load_series(
            asset_id, reindex=None, end_date=end_date)

    else:
        sr = pd.Series()


    wt = TimingWt(sr)
    filtered_data = wt.wavefilter(sr, wavelet_filter_num)
    filtered_data = filtered_data[filtered_data.index >= begin_date]
    filtered_data = filtered_data.loc[reindex]

    return filtered_data


def load_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):

    prefix = asset_id[0:2]
    if prefix.isdigit():
        xtype = int(asset_id) // 10000000
        if xtype == 1:
            #
            # 基金池资产
            #
            asset_id = int(asset_id) % 10000000
            (pool_id, category) = (asset_id // 100, asset_id % 100)
            ttype = pool_id / 10000
            sr = asset_ra_pool_nav.load_series(
                pool_id, category, ttype, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif xtype == 3:
            #
            # 基金池资产
            #
            sr = base_ra_fund_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif xtype == 4:
            #
            # 修型资产
            #
            sr = asset_rs_reshape_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif xtype == 12:
            #
            # 指数资产
            #
            sr = base_ra_index_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif xtype == 'ERI':

            sr = base_exchange_rate_index_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        else:
            sr = pd.Series()
    else:
        if prefix == 'AP':
            #
            # 基金池资产
            #
            sr = asset_ra_pool_nav.load_series(
                asset_id, 0, 9, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'FD':
            #
            # 基金资产
            #
            sr = base_ra_fund_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'RS':
            #
            # 修型资产
            #
            sr = asset_rs_reshape_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'IX':
            #
            # 指数资产
            #
            sr = base_ra_index_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'ER':

            sr = base_exchange_rate_index_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        else:
            sr = pd.Series()

    return sr



@markowitz.command()
@click.option('--id', 'optid', help=u'ids of markowitz to update')
@click.option('--type', 'opttype', default='8,9', help=u'which type to run')
@click.option('--risk', 'optrisk', default='10,1,2,3,4,5,6,7,8,9', help=u'which risk to calc, [1-10]')
@click.option('--append/--no-append', 'optappend', default=False, help=u'append pos or not')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.option('--start-date', 'sdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'edate', help=u'end date to calc')
@click.option('--cpu-count', 'optcpu', type=int, default=0, help=u'how many cpu to use, (0 for all available)')
@click.pass_context
def pos(ctx, optid, optlist, opttype, optrisk, optappend, sdate, edate, optcpu):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        markowitzs = [s.strip() for s in optid.split(',')]
    else:
        if 'markowitz' in ctx.obj:
            markowitzs = [str(ctx.obj['markowitz'])]
        else:
            markowitzs = None

    xtypes = [s.strip() for s in opttype.split(',')]

    df_markowitz = asset_mz_markowitz.load(markowitzs, xtypes)
    if optlist:
        df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_markowitz, headers='keys', tablefmt='psql'))
        return 0

    for _, markowitz in df_markowitz.iterrows():
        pos_update_alloc(markowitz, optrisk, optappend, sdate, edate, optcpu)

def pos_update_alloc(markowitz, optrisk, optappend, sdate, edate, optcpu):
    risks =  [("%.2f" % (float(x)/ 10.0)) for x in optrisk.split(',')];
    df_alloc = asset_mz_markowitz_alloc.where_markowitz_id(markowitz['globalid'], risks)
    #print df_alloc

    for _, alloc in df_alloc.iterrows():
        pos_update(markowitz, alloc, optappend, sdate, edate, optcpu)

    click.echo(click.style("markowitz allocation complement! instance id [%s]" % (markowitz['globalid']), fg='green'))

def pos_update(markowitz, alloc, optappend, sdate, edate, optcpu):
    markowitz_id = alloc['globalid']
    #
    # 加载资产
    #
    df_asset = asset_mz_markowitz_asset.load([markowitz_id])
    #print df_asset
    df_asset.set_index(['mz_markowitz_asset_id'], inplace=True)

    df_asset = df_asset[['mz_upper_limit', 'mz_lower_limit', 'mz_sum1_limit', 'mz_sum2_limit']];
    df_asset = df_asset.rename(columns={'mz_upper_limit': 'upper', 'mz_lower_limit': 'lower', 'mz_sum1_limit': 'sum1', 'mz_sum2_limit': 'sum2'})
    assets = df_asset.T.to_dict()
    #
    # 加载参数
    #
    df_argv = asset_mz_markowitz_argv.load([markowitz_id])
    df_argv.reset_index(level=0, inplace=True)
    argv = df_argv['mz_value'].to_dict()

    lookback = int(argv.get('lookback', '26'))
    adjust_period = int(argv.get('adjust_period', 1))
    wavelet_filter_num = int(argv.get('allocate_wavelet', 0))
    turnover = float(argv.get('allocate_turnover_filter', 0))
    bound_id = argv.get('asset_bound', None)
    if bound_id is not None:
        bounds = AssetBound.load_asset_bounds(bound_id)
    else:
        bounds = None


    algo = alloc['mz_algo'] if alloc['mz_algo'] != 0 else markowitz['mz_algo']

    #print optappend, sdate, markowitz_id

    #load df old pos
    df_pos_old = asset_mz_markowitz_pos.load_raw(markowitz_id)
    #print df_pos_old.tail()
    if len(df_pos_old) <= 4:
        optappend = False
    elif optappend:
        sdate = df_pos_old.index[-4]
        df_pos_old = df_pos_old.iloc[:-1,]
    else:
        pass


    if algo == 1:

        #均配
        optappend = False
        df = average_days(sdate, edate, assets)
        if 'return' in df.columns:
            df.drop(['return', 'risk', 'sharpe'], axis=1, inplace=True)
        for asset in df.columns:
            if asset.startswith('MZ'):
                mz_pos_df = asset_mz_markowitz_pos.load_raw(asset)
                asset_pos = df[asset]
                dates = list(set(mz_pos_df.index | df.index))
                dates.sort()
                asset_pos = asset_pos.reindex(dates)
                asset_pos.fillna(method = 'pad', inplace=True)
                asset_pos.fillna(0.0, inplace=True)
                mz_pos_df = mz_pos_df.mul(asset_pos, axis = 0).fillna(method = 'pad')
                df = df.drop(asset, axis = 1)
                df = pd.concat([df, mz_pos_df], axis = 1, join_axes = [mz_pos_df.index])
        df = df.T
        df = df.groupby(df.index).sum()
        df = df.T
        #trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback)
        #df = df.reindex(trade_date)
        #df = df.fillna(method = 'pad')
        #df = df.dropna()

    elif algo == 2:

        #马科维兹
        #df = markowitz_days(
        #    sdate, edate, assets,
        #    label='markowitz', lookback=lookback, adjust_period=adjust_period, bootstrap=None, cpu_count=optcpu, wavelet = False)
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzAllocate('ALC.000001', assets, trade_date, lookback, bound = bounds)
        df = allocate.allocate()

    elif algo == 3:

        #马科维兹boot
        #df = markowitz_days(
        #    sdate, edate, assets,
        #    label='markowitz', lookback=lookback, adjust_period=adjust_period, bootstrap=0, cpu_count=optcpu, wavelet = False)
        #print df.tail()
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzBootAllocate('ALC.000001', assets, trade_date, lookback, bound = bounds)
        df = allocate.allocate()

    elif algo == 4:

        #滤波
        #df = markowitz_days(
        #    sdate, edate, assets,
        #    label='markowitz', lookback=lookback, adjust_period=adjust_period, bootstrap=None, cpu_count=optcpu, wavelet = True, wavelet_filter_num = wavelet_filter_num)
        wavelet_filter_num = int(argv.get('allocate_wavelet', 0))
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , WaveletAsset(asset_id, wavelet_filter_num)) for asset_id in list(assets.keys())])
        allocate = MzAllocate('ALC.000001', assets, trade_date, lookback, bound = bounds)
        df = allocate.allocate()

    elif algo == 5:

        #blacklitterman
        #df = markowitz_days(
        #    sdate, edate, assets,
        #    label='markowitz', lookback=lookback, adjust_period=adjust_period, bootstrap=0, cpu_count=optcpu, blacklitterman = True, wavelet = False, wavelet_filter_num = wavelet_filter_num, markowitz_id = markowitz_id)
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        views = {}
        for asset_id in assets.keys():
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)
        allocate = MzBootBlAllocate('ALC.000001', assets, views, trade_date, lookback, bound = bounds)
        df = allocate.allocate()

    elif algo == 6:

        #blacklitterman 滤波
        #df = markowitz_days(
        #    sdate, edate, assets,
        #    label='markowitz', lookback=lookback, adjust_period=adjust_period, bootstrap=None, cpu_count=optcpu, blacklitterman = True, wavelet = True, wavelet_filter_num = wavelet_filter_num,  markowitz_id = markowitz_id)
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        wavelet_filter_num = int(argv.get('allocate_wavelet', 0))
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))
        assets = dict([(asset_id , WaveletAsset(asset_id, wavelet_filter_num)) for asset_id in list(assets.keys())])
        views = {}
        for asset_id in assets.keys():
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)
        allocate = MzBlAllocate('ALC.000001', assets, views, trade_date, lookback, bound = bounds)
        df = allocate.allocate()

    elif algo == 7:

        #blacklitterman和滤波blacklitterman均配
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        wavelet_filter_num = int(argv.get('allocate_wavelet', 0))
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))

        views = {}
        for asset_id in assets.keys():
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)

        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzBootBlAllocate('ALC.000001', assets, views, trade_date, lookback, bound = bounds)
        bootbl_df = allocate.allocate()

        assets = dict([(asset_id , WaveletAsset(asset_id, wavelet_filter_num)) for asset_id in list(assets.keys())])
        allocate = MzBlAllocate('ALC.000001', assets, views, trade_date, lookback, bound = bounds)
        waveletbl_df = allocate.allocate()

        bootbl_df = bootbl_df[waveletbl_df.columns]

        df = (bootbl_df + waveletbl_df) / 2


    elif algo == 8:

        #滤波下方差
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        wavelet_filter_num = int(argv.get('allocate_wavelet', 0))
        assets = dict([(asset_id , WaveletAsset(asset_id, wavelet_filter_num)) for asset_id in list(assets.keys())])
        allocate = MzBootDownRiskAllocate('ALC.000001', assets, trade_date, lookback, bound = bounds)
        df = allocate.allocate()

    elif algo == 9:

        #固定波动率配置
        upper_risk = float(argv.get('upper_risk', 0.015))
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzFixRiskBootAllocate('ALC.000001', assets, trade_date, lookback, upper_risk, bound = bounds)
        df = allocate.allocate()

    elif algo == 10:

        #固定波动率BL配置
        upper_risk = float(argv.get('upper_risk'))
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))

        views = {}
        for asset_id in assets.keys():
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)

        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzFixRiskBootBlAllocate('ALC.000001', assets, views, trade_date, lookback, risk=upper_risk, bound=bounds)
        df = allocate.m_allocate()

    elif algo == 11:

        #固定波动率滤波配置
        upper_risk = float(argv.get('upper_risk', 0.018))
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        wavelet_filter_num = int(argv.get('allocate_wavelet', 0))
        wavelet_assets = dict([(asset_id , WaveletAsset(asset_id, wavelet_filter_num)) for asset_id in list(assets.keys())])
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzFixRiskBootWaveletAllocate('ALC.000001', assets, wavelet_assets, trade_date, lookback, upper_risk, bound = bounds, period = adjust_period)
        df = allocate.allocate()

    elif algo == 12:

        #固定波动率滤波Bl配置
        upper_risk = float(argv.get('upper_risk', 0.018))
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        wavelet_filter_num = int(argv.get('allocate_wavelet', 0))
        wavelet_assets = dict([(asset_id , WaveletAsset(asset_id, wavelet_filter_num)) for asset_id in list(assets.keys())])
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))
        views = {}
        for asset_id in assets.keys():
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzFixRiskBootWaveletBlAllocate('ALC.000001', assets, wavelet_assets, views, trade_date, lookback, risk=upper_risk, bound=bounds)
        df = allocate.m_allocate()


    elif algo == 13:

        #multifactor index
        trade_date = ATradeDate.trade_date(begin_date = sdate, end_date = edate, lookback=lookback+1)
        bound = AssetBound('asset_bound_default', [asset_id for asset_id in assets.keys()], upper = 0.1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = FactorValidAllocate('ALC.000001', assets, trade_date, lookback, bound = bound, period = period)
        df = allocate.allocate()

    elif algo == 14:

        #single factor index
        lookback = int(argv.get('lookback'))
        # period = int(argv.get('period'))
        period = 1
        factor_num = int(argv.get('factor_num'))
        factor_loc = int(argv.get('factor_loc'))
        factor_end = int(argv.get('factor_end'))
        target = np.zeros(factor_num)
        target[factor_loc] = 1
        target = target * factor_end
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, end_date = edate, lookback=lookback+1)
        allocate = FactorIndexAllocate('ALC.000001', trade_date, lookback, period = period, target = target)
        df = allocate.allocate()

    elif algo == 15:

        # A Layer Fix Risk Boot Bl

        upper_risk = float(argv.get('upper_risk', 0.015))
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))

        asset_ids = list(assets.keys())
        asset_ids.append('ALayer')
        views = {}
        for asset_id in asset_ids:
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)

        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzALayerFixRiskBootBlAllocate('ALC.000001', assets, views, trade_date, lookback, risk=upper_risk, bound=bounds)
        df = allocate.m_allocate()

        print(df.tail())

    elif algo == 16:

        #use momentum algo to select valid factor

        lookback = int(argv.get('lookback', '26'))
        alloc_num = int(argv.get('alloc_num', '5'))
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = SingleValidFactorAllocate('ALC.000001', assets, trade_date, lookback, alloc_num, bound = bounds)
        df = allocate.allocate()

    elif algo == 17:

        #monetary allocate

        lookback = int(argv.get('lookback', '52'))
        trade_date = ATradeDate.week_trade_date(begin_date=sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in MonetaryAllocate.all_monetary_fund_globalid()])
        allocate = MonetaryAllocate('ALC.000001', assets, trade_date, lookback, period=1)
        df = allocate.allocate()

    elif algo == 18:

        #cppi allocate

        lookback = 365
        trade_date = ATradeDate.trade_date(begin_date=sdate, lookback=lookback+1)
        assets = dict([(asset_id, Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = CppiAllocate('ALC.000001', assets, trade_date, lookback, bound=bounds, period=1)
        df = allocate.allocate_cppi()

    elif algo == 19:

        # Risk Parity

        lookback = int(argv.get('lookback', '28'))

        asset_ids = list(assets.keys())
        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)

        assets = dict([(asset_id , Asset(asset_id)) for asset_id in asset_ids])

        allocate = RpAllocate('ALC.000001', assets, trade_date, lookback, bound=bounds)
        df = allocate.m_allocate()

    elif algo == 20:

        # Layer Fix Risk Boot Bl

        upper_risk = float(argv.get('upper_risk', 0.015))
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))
        data_period = argv.get('data_period', 'day')

        asset_ids = list(assets.keys())
        views = {}
        for asset_id in asset_ids:
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)

        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzLayerFixRiskBootBlAllocate('ALC.000001', assets, views, trade_date, lookback, data_period=data_period, risk=upper_risk, bound=bounds)
        df = allocate.m_allocate()

    elif algo == 21:

        # Layer Fix Risk Cov Smooth Boot Bl

        lookback = int(argv.get('lookback', '26'))
        smooth = int(argv.get('smooth', '0'))
        upper_risk = float(argv.get('upper_risk'))
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))
        data_period = argv.get('data_period', 'week')
        cov_algo = argv.get('cov_algo')

        asset_ids = list(assets.keys())
        views = {}
        for asset_id in asset_ids:
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)

        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzLayerFixRiskCovSmoothBootBlAllocate('ALC.000001', assets, views, trade_date, lookback, smooth=smooth, data_period=data_period, risk=upper_risk, cov_algo=cov_algo, bound=bounds)
        df = allocate.m_allocate()

    elif algo == 22:

        # Layer Fix Risk Cov Smooth Boot Bl

        lookback = int(argv.get('lookback', '26'))
        smooth = int(argv.get('smooth', '0'))
        upper_risk = float(argv.get('upper_risk'))
        view_df = View.load_view(argv.get('bl_view_id'))
        confidence = float(argv.get('bl_confidence'))
        data_period = argv.get('data_period', 'week')
        cov_algo = argv.get('cov_algo')
        '''

        asset_ids = list(assets.keys())
        views = {}
        for asset_id in asset_ids:
            views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = confidence) if asset_id in view_df.columns else View(None, asset_id, confidence = confidence)

        trade_date = ATradeDate.week_trade_date(begin_date = sdate, lookback=lookback+1)
        assets = dict([(asset_id , Asset(asset_id)) for asset_id in list(assets.keys())])
        allocate = MzLayerFixRiskCovSampleBlAllocate('ALC.000001', assets, views, trade_date, lookback, smooth=smooth, data_period=data_period, risk=upper_risk, cov_algo=cov_algo, bound=bounds)
        df = allocate.m_allocate()


        allocate = RaViewIndexAllocate('ALC.000001', assets, trade_date, lookback)
        df = allocate.allocate()
        '''
        pass
    else:
        click.echo(click.style("\n unknow algo %d for %s\n" % (algo, markowitz_id), fg='red'))
        return;

    if 'return' in df.columns:
        df_sharpe = df[['return', 'risk', 'sharpe']].copy()
        df.drop(['return', 'risk', 'sharpe'], axis=1, inplace=True)


    #if optappend:
    #    df = pd.concat([df_pos_old, df]).fillna(0.0)


    db = database.connection('asset')
    metadata = MetaData(bind=db)
    mz_markowitz_pos    = Table('mz_markowitz_pos', metadata, autoload=True)
    mz_markowitz_nav    = Table('mz_markowitz_nav', metadata, autoload=True)

    #
    # 处理替换
    #
    mz_markowitz_pos.delete(mz_markowitz_pos.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz_nav.delete(mz_markowitz_nav.c.mz_markowitz_id == markowitz_id).execute()

    #
    # 导入数据: markowitz_pos
    #
    df = df.round(4)             # 四舍五入到万分位

    # 不需要进行四周平滑的algo id
    no_rolling_algos = [1, 16, 17, 18, 19, 22]
    if algo in no_rolling_algos:
        pass
    else:
    #每四周做平滑
        df = df.rolling(window = 4, min_periods = 1).mean()

    if optappend:
        df = df.iloc[3:,:]


    df[df.abs() < 0.0009999] = 0 # 过滤掉过小的份额
    df = df.apply(npu.np_pad_to, raw=True, axis=1) # 补足缺失
    #df = DFUtil.filter_same_with_last(df)          # 过滤掉相同
    if turnover >= 0.01:
        df = DFUtil.filter_by_turnover(df, turnover)   # 基于换手率进行规律

    if optappend:
        df = pd.concat([df_pos_old, df]).fillna(0.0)

    print(df.tail())


    df.index.name = 'mz_date'
    df.columns.name='mz_markowitz_asset'

    # 计算原始资产仓位
    raw_ratios = {}
    raw_assets = {}
    for asset_id in df.columns:
        raw_assets[asset_id] = asset_id
        raw_ratios[asset_id] = df[asset_id]

    df_raw_ratio = pd.DataFrame(raw_ratios, columns=df.columns)
    df_raw_asset = pd.DataFrame(raw_assets, index=df.index, columns=df.columns)

    df_tosave = pd.concat({'mz_markowitz_ratio': df, 'mz_asset_id':df_raw_asset, 'mz_ratio': df_raw_ratio}, axis=1)

    # index
    df_tosave['mz_markowitz_id'] = markowitz_id
    df_tosave = df_tosave.reset_index().set_index(['mz_markowitz_id', 'mz_date'])

    # unstack
    df_tosave = df_tosave.stack()
    #print df_tosave.head()
    df_tosave = df_tosave.loc[(df_tosave['mz_ratio'] > 0) | (df_tosave['mz_markowitz_ratio'] > 0)]

    # save
    asset_mz_markowitz_pos.save(markowitz_id, df_tosave)

    return 0

@markowitz.command()
@click.option('--id', 'optid', help=u'ids of markowitz to update')
@click.option('--type', 'opttype', default='8,9', help=u'which type to run')
@click.option('--risk', 'optrisk', default='10,1,2,3,4,5,6,7,8,9', help=u'which risk to calc, [1-10]')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def nav(ctx, optid, opttype, optrisk, optlist):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        markowitzs = [s.strip() for s in optid.split(',')]
    else:
        if 'markowitz' in ctx.obj:
            markowitzs = [str(ctx.obj['markowitz'])]
        else:
            markowitzs = None

    xtypes = [s.strip() for s in opttype.split(',')]

    if markowitzs is not None:
        df_markowitz = asset_mz_markowitz.load(markowitzs)
    else:
        df_markowitz = asset_mz_markowitz.load(markowitzs, xtypes)

    if optlist:
        df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_markowitz, headers='keys', tablefmt='psql'))
        return 0

    with click.progressbar(
            df_markowitz.iterrows(), len(df_markowitz.index), label='update nav'.ljust(30),
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, markowitz in bar:
            # bar.update(1)
            nav_update_alloc(markowitz, optrisk)

def nav_update_alloc(markowitz, optrisk):
    risks =  [("%.2f" % (float(x)/ 10.0)) for x in optrisk.split(',')];
    df_alloc = asset_mz_markowitz_alloc.where_markowitz_id(markowitz['globalid'], risks)

    for _, alloc in df_alloc.iterrows():
        nav_update(markowitz, alloc)

def nav_update(markowitz, alloc):
    gid = alloc['globalid']
    # 加载仓位信息
    df_pos = asset_mz_markowitz_pos.load(gid)

    # 加载资产收益率
    min_date = df_pos.index.min()
    #max_date = df_pos.index.max()
    max_date = (datetime.now() - timedelta(days=1)) # yesterday

    data = {}
    for asset_id in df_pos.columns:
        try:
            data[asset_id] = Asset.load_nav_series(asset_id, begin_date=min_date, end_date=max_date)
        except:
            print('not have asset : ' + asset_id)
    df_nav = pd.DataFrame(data).fillna(method='pad')
    df_inc  = df_nav.pct_change().fillna(0.0)
    df_pos = df_pos[df_inc.columns]

    # 计算复合资产净值
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_pos, result_col='portfolio')

    df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'mz_nav'}).copy()
    df_result.index.name = 'mz_date'
    df_result['mz_inc'] = df_result['mz_nav'].pct_change().fillna(0.0)
    df_result['mz_markowitz_id'] = gid
    df_result = df_result.reset_index().set_index(['mz_markowitz_id', 'mz_date'])

    asset_mz_markowitz_nav.save(gid, df_result)

@markowitz.command()
@click.option('--id', 'optid', help=u'ids of markowitz to update')
@click.option('--type', 'opttype', default='8,9', help=u'which type to run')
@click.option('--risk', 'optrisk', default='10,1,2,3,4,5,6,7,8,9', help=u'which risk to calc, [1-10]')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def turnover(ctx, optid, opttype, optrisk, optlist):
    ''' calc pool turnover and inc
    '''
    if optid is not None:
        markowitzs = [s.strip() for s in optid.split(',')]
    else:
        if 'markowitz' in ctx.obj:
            markowitzs = [str(ctx.obj['markowitz'])]
        else:
            markowitzs = None

    xtypes = [s.strip() for s in opttype.split(',')]

    if markowitzs is not None:
        df_markowitz = asset_mz_markowitz.load(markowitzs)
    else:
        df_markowitz = asset_mz_markowitz.load(markowitzs, xtypes)

    if optlist:
        df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_markowitz, headers='keys', tablefmt='psql'))
        return 0

    data = []
    with click.progressbar(
            df_markowitz.iterrows(), length=len(df_markowitz.index), label= 'update turnover'.ljust(30),
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, markowitz in bar:
            # bar.update(1)
            segments = turnover_update_alloc(markowitz, optrisk)
            data.extend(segments)

    headers = ['markowitz', 'turnover(%)']
    print(tabulate(data, headers=headers, tablefmt="psql"))
    # print(tabulate(data, headers=headers, tablefmt="fancy_grid"))
    # print(tabulate(data, headers=headers, tablefmt="grid"))

def turnover_update_alloc(markowitz, optrisk):
    risks =  [("%.2f" % (float(x)/ 10.0)) for x in optrisk.split(',')];
    df_alloc = asset_mz_markowitz_alloc.where_markowitz_id(markowitz['globalid'], risks)

    data = []
    for _, alloc in df_alloc.iterrows():
        turnover = turnover_update(markowitz, alloc)
        data.append((alloc['globalid'], "%6.2f" % (turnover * 100)))

    return data

def turnover_update(markowitz, alloc):
    gid = alloc['globalid']
    # 加载仓位信息
    df = asset_mz_markowitz_pos.load(gid, use_markowitz_ratio=False)

    # 计算宽口换手率
    sr_turnover = DFUtil.calc_turnover(df)

    criteria_id = 6
    df_result = sr_turnover.to_frame('mz_value')
    df_result['mz_markowitz_id'] = gid
    df_result['mz_criteria_id'] = criteria_id
    df_result = df_result.reset_index().set_index(['mz_markowitz_id', 'mz_criteria_id', 'mz_date'])
    asset_mz_markowitz_criteria.save(gid, criteria_id,  df_result)

    total_turnover = sr_turnover.sum()

    return total_turnover

@markowitz.command()
@click.option('--id', 'optid', help=u'ids of markowitz to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.option('--exec/--no-exec', 'optexec', default=False, help=u'list instance to update')
@click.pass_context
def delete(ctx, optid, optlist, optexec):
    ''' delete markowitz instance
    '''
    if optid is not None:
        markowitzs = [s.strip() for s in optid.split(',')]
    else:
        markowitzs = None

    df_markowitz = asset_mz_markowitz.load(markowitzs)

    if optlist:

        df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_markowitz, headers='keys', tablefmt='psql'))
        return 0

    if optid is None or not optexec:
         click.echo(click.style("\nboth --id and --exec is required to perform delete\n", fg='red'))
         return 0

    data = []
    with click.progressbar(
            df_markowitz.iterrows(), length=len(df_markowitz.index), label= 'delete markowitz'.ljust(30),
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, markowitz in bar:
            perform_delete(markowitz)

def perform_delete(markowitz):
    markowitz_id = markowitz['globalid']

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    mz_markowitz        = Table('mz_markowitz', metadata, autoload=True)
    mz_markowitz_asset  = Table('mz_markowitz_asset', metadata, autoload=True)
    mz_markowitz_pos    = Table('mz_markowitz_pos', metadata, autoload=True)
    mz_markowitz_nav    = Table('mz_markowitz_nav', metadata, autoload=True)
    mz_markowitz_sharpe = Table('mz_markowitz_sharpe', metadata, autoload=True)

    #
    # 处理删除
    #

    mz_markowitz.delete(mz_markowitz.c.globalid == markowitz_id).execute()
    mz_markowitz_asset.delete(mz_markowitz_asset.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz_pos.delete(mz_markowitz_pos.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz_nav.delete(mz_markowitz_nav.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz_sharpe.delete(mz_markowitz_sharpe.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz.delete(mz_markowitz.c.globalid == markowitz_id).execute()

@markowitz.command()
@click.option('--src', 'optsrc', help=u'src id of markowitz to copy from')
@click.option('--dst', 'optdst', help=u'dst id of markowitz to copy to')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def copy(ctx, optsrc, optdst, optlist):
    ''' create new markowitz by copying  existed one
    '''
    if optsrc is not None:
        markowitzs = [optsrc]
    else:
        markowitzs = None

    df_markowitz = asset_mz_markowitz.load(markowitzs)

    if optlist:

        df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_markowitz, headers='keys', tablefmt='psql'))
        return 0

    if optsrc  is None or optdst is None:
        click.echo(click.style("\n both --src-id  and --dst-id is required to perform copy\n", fg='red'))
        return 0

    #
    # copy mz_markowitz
    #
    df_markowitz['globalid'] = optdst
    df_markowitz.set_index(['globalid'], inplace=True)
    asset_mz_markowitz.save(optdst, df_markowitz)

    #
    # copy mz_markowitz_alloc
    #
    df_markowitz_alloc = asset_mz_markowitz_alloc.load(optsrc)
    df_markowitz_alloc.reset_index(inplace=True)

    df_markowitz_alloc['mz_markowitz_id'] = optdst
    df_markowitz_alloc['old'] = df_markowitz_alloc['globalid']
    sr_tmp = df_markowitz_alloc['mz_markowitz_id'].str[:len(optdst) - 1]
    df_markowitz_alloc['globalid'] = sr_tmp.str.cat((df_markowitz_alloc['mz_risk'] * 10 % 10).astype(int).astype(str))

    df_xtab = df_markowitz_alloc[['globalid', 'old']].copy()

    df_markowitz_alloc.drop(['old'], axis=1, inplace=True)

    df_markowitz_alloc.set_index(['globalid'], inplace=True)
    asset_mz_markowitz_alloc.save(optdst, df_markowitz_alloc)

    #
    # copy mz_markowitz_argv
    #
    df_markowitz_argv = asset_mz_markowitz_argv.load(df_xtab['old'])
    df_markowitz_argv.reset_index(inplace=True)

    df_markowitz_argv = df_markowitz_argv.merge(df_xtab, left_on='mz_markowitz_id', right_on = 'old')
    df_markowitz_argv['mz_markowitz_id'] = df_markowitz_argv['globalid']
    df_markowitz_argv.drop(['globalid', 'old'], inplace=True, axis=1)
    df_markowitz_argv = df_markowitz_argv.set_index(['mz_markowitz_id', 'mz_key'])

    asset_mz_markowitz_argv.save(df_xtab['globalid'], df_markowitz_argv)

    #
    # copy mz_markowitz_asset
    #
    df_markowitz_asset = asset_mz_markowitz_asset.load(df_xtab['old'])
    # df_markowitz_asset.reset_index(inplace=True)

    df_markowitz_asset = df_markowitz_asset.merge(df_xtab, left_on='mz_markowitz_id', right_on = 'old')

    df_markowitz_asset['mz_markowitz_id'] = df_markowitz_asset['globalid']
    df_markowitz_asset.drop(['globalid', 'old'], inplace=True, axis=1)
    df_markowitz_asset = df_markowitz_asset.set_index(['mz_markowitz_id', 'mz_markowitz_asset_id'])

    asset_mz_markowitz_asset.save(df_xtab['globalid'], df_markowitz_asset)
