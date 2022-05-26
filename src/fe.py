import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import adjust_price


def preprocess(dfs):
    print('cleaning data...', end='')
    df_price, df_fin, df_list = dfs['price'], dfs['financial'], dfs['list']

    # pre-processing of stock price

    # fill nan ex-dividend
    df_price.ExpectedDividend.fillna(0, inplace=True)
    # TODO: fill numerical features(Close) ?

    # financials
    df_fin.replace('－', 0.0, inplace=True)
    df_fin.iloc[:, 12:] = df_fin.iloc[:, 12:].astype('float32').fillna(0)
    print('Finished!')
    return dfs


def create_features(dfs):
    print('Creating new features for stock price...', end='')
    df_price, df_fn, df_list = dfs['price'], dfs['financial'], dfs['list']

    price_cols = ['Close']

    # df_price[price_cols] = df_price[price_cols].replace(0.0, np.nan)

    # add high/low
    df_price["diff_high_low"] = (
        np.log1p(df_price["High"]) / np.log1p(df_price["Low"])
        ).astype("float16")
        
    log_price = df_price[price_cols].apply(np.log1p)
    diff_log_price = log_price.diff()
    diff_log_price['SecuritiesCode'] = df_price.SecuritiesCode

    # add new features
    fea_price = [df_price]

    group_code = df_price.groupby("SecuritiesCode")
    price_gb_code = df_price.groupby("SecuritiesCode")[price_cols]
    diff_log_price_gb_code = diff_log_price.groupby("SecuritiesCode")[price_cols]

    for span in [20, 40, 60, 120]:
        pct_change = (
            price_gb_code.pct_change(span)
                .add_prefix(f"feat_pct{span}_")
                .astype('float16')
        )
        volatility = (
            diff_log_price_gb_code.rolling(span, min_periods=1)
                .std()
                .reset_index("SecuritiesCode", drop=True)
        )
        volatility = (
            (log_price / volatility).add_prefix(f"feat_vl{span}_")
                .astype('float16')
        )
        ma = (
            price_gb_code.rolling(span, min_periods=1)
                .mean()
                .reset_index("SecuritiesCode", drop=True)
        )
        ma = (df_price[price_cols] / ma).add_prefix(f"feat_rmr{span}_").astype('float16')
        diff_roll = (
            group_code.diff_high_low
                .rolling(span, min_periods=1)
                .mean().
                rename(f"feat_wd{span}")
                .reset_index("SecuritiesCode", drop=True)
                .astype('float16')
        )
        fea_price += [pct_change, volatility, ma, diff_roll]

    df_price_fe = pd.concat(fea_price, axis=1)
    df_price_fe = adjust_price(df_price_fe) # add adjusted close value
    print('Finished!')

    # financial features
    print('Creating new financial features...')
    acc_cols = ['NetSales', 'Profit', 'OperatingProfit', 'OrdinaryProfit']
    df_fn[acc_cols] = df_fn[acc_cols].replace('－', 0).astype('float32')
    # drop irregular financial quarter
    df_fn = df_fn[~df_fn.TypeOfCurrentPeriod.isin(['4Q', '5Q', np.nan])].reset_index(drop=True)
    df_fn['quarter'] = df_fn.TypeOfCurrentPeriod.map({"1Q": 1, "2Q": 2, "3Q": 3, "FY":4})
    df_fn['FiscalYear'] = df_fn.CurrentFiscalYearEndDate.str.split('-', expand=True)[0].astype('int16')

    fp_ymd = df_fn.CurrentPeriodEndDate.str.split('-', expand=True).astype('float32')
    df_fn['fp_month'] = (fp_ymd[0] - 2016) * 12 + fp_ymd[1]
    df_quarter = (
        df_fn[~(df_fn.CurrentPeriodEndDate.isna())]
            .groupby(['SecuritiesCode', 'CurrentPeriodEndDate'])
            .nth(-1)
    )

    df_quarter['quarter_span'] = df_quarter.groupby('SecuritiesCode').fp_month.diff().fillna(3)
    df_fn = df_fn.join(df_quarter.quarter_span, on=['SecuritiesCode', 'CurrentPeriodEndDate'])
    year_span = (df_quarter.groupby(['SecuritiesCode', 'FiscalYear']).quarter_span.sum()).rename('year_span')
    df_fn = df_fn.join(year_span, on=['SecuritiesCode', 'FiscalYear'])
    df_diff = df_quarter.groupby(['SecuritiesCode', 'FiscalYear'])[acc_cols].diff().fillna(
        df_quarter[acc_cols] / df_quarter.quarter.values[:, None]
    )
    df_diff /= (df_quarter["quarter_span"].values[:, None] / 3)

    df_ma = df_diff[acc_cols].rolling(4, min_periods=1).mean()
    df_diff = df_diff.add_prefix('diff_')
    df_ma = df_ma.add_prefix('ma_')

    df_fn = df_fn.merge(df_diff, on=['SecuritiesCode', 'CurrentPeriodEndDate'], how='left')
    df_fn = df_fn.merge(df_ma, on=['SecuritiesCode', 'CurrentPeriodEndDate'], how='left')

    amount_cols = ['TotalAssets', 'Equity']

    amount_cols += ["diff_" + col for col in acc_cols]
    amount_cols += ["ma_" + col for col in acc_cols]
    amount_cols += acc_cols

    key_cols = ["SecuritiesCode", "FiscalYear", "TypeOfCurrentPeriod"]
    df_last_year = df_fn.groupby(key_cols)[amount_cols].nth(-1).reset_index()
    df_last_year.FiscalYear += 1
    df_last_fn = df_fn[key_cols].merge(df_last_year, on=key_cols, how='left')
    df_last_ratio = (df_fn[amount_cols] - df_last_fn[amount_cols]) / df_last_fn[amount_cols]
    df_last_ratio = df_last_ratio.add_prefix('feat_ratio_')

    df_fn = pd.concat([df_fn, df_last_ratio], axis=1)
    df_fn.Date = pd.to_datetime(df_fn.Date)

    def align_quarter(df, table):
        res = []
        for code, date in tqdm(zip(df.SecuritiesCode.values, df.Date.values), total=len(df)):
            flag = False
            ts = table.get(code, None)
            if not ts:
                res.append(date)
                continue
            for t in ts:
                if (date < t and date > t - pd.tseries.offsets.DateOffset(90)):
                    res.append(t)
                    flag = True
                    break
            if not flag:
                res.append(date)
        return res

    df_price_fe.Date = pd.to_datetime(df_price_fe.Date)
    df_fn.CurrentPeriodEndDate = pd.to_datetime(df_fn.CurrentPeriodEndDate)

    print('Adding CurrentPeriodEndDate to price frame...')
    table = df_fn.groupby('SecuritiesCode')['CurrentPeriodEndDate'].apply(list).to_dict()
    df_price_fe['CurrentPeriodEndDate'] = align_quarter(df_price_fe, table)

    dfs['price'] = df_price_fe
    dfs['financial'] = df_fn
    print('FE Finished!')

    return dfs