import numpy as np
import pandas as pd
import lightgbm as lgb

from data import read_data
from fe import preprocess, create_features
from utils import PurgedGroupTimeSeriesSplit


def train():
    ...

def main():
    file_paths = {
        'price': 'input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv',
        'list': 'input/jpx-tokyo-stock-exchange-prediction/stock_list.csv',
        'financial': 'input/jpx-tokyo-stock-exchange-prediction/train_files/financials.csv'
    }
    
    dfs = create_features(
        preprocess(read_data(file_paths))
    )

    df = dfs['price'].merge(dfs['financial'], on=['SecuritiesCode', 'CurrentPeriodEndDate'], how='left')
    df = df.drop('Date_y', axis=1).rename(columns={'Date_x': 'Date'})

    cv = PurgedGroupTimeSeriesSplit(n_splits=10, max_train_group_size=200, max_test_group_size=50, group_gap=30)
    y = df.Target
    X = df.drop(['RowID', 'SecuritiesCode', 'Target'], axis=1)

    for 

if __name__ == '__main__':
    main()