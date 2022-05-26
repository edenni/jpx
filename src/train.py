import logging
import sys

import numpy as np
import pandas as pd
# import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

from data import read_data
from fe import preprocess, create_features
from utils import PurgedGroupTimeSeriesSplit, add_rank, calc_spread_return_sharpe

FOLDS = 5
GROUP_GAP = 20
MAX_TEST_GROUP_SIZE = 60
MAX_TRAIN_GROUP_SIZE = 200



params = {
    'objective': 'reg:squarederror',
    'eval_metric': "rmse",
    'max_depth': 8,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'learning_rate': 0.05,
}

def objective_wrapper(cv, X, y, groups):
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': "rmse",
            # 'tree_method': 'gpu_hist',
            'learning_rate': 0.01,
            'max_depth': trial.suggest_int('max_depth', 6, 9),
            'subsample': trial.suggest_uniform('subsample', 0.4, 0.8),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 0.8),
            'lambda': trial.suggest_uniform('lambda', 1, 2),
            'max_bin': trial.suggest_uniform('max_bin', 128, 256),
        }

        return train(cv, X, y, groups, params)
    return objective

def train(cv, X, y, groups):

    print('Start training...')
    res = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_train = X_train.drop('Date', axis=1)
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        X_val_nodate = X_val.drop('Date', axis=1)
        dval = xgb.DMatrix(X_val_nodate, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'val')]

        model = xgb.train(params, dtrain, 1000, verbose_eval=False, evals=watchlist, early_stopping_rounds=30)

        pred = model.predict(dval)

        df_val = X_val.copy()
        df_val['pred'] = pred
        df_val['Target'] = y_val

        sharpe, _ = calc_spread_return_sharpe(add_rank(df_val))
        res.append(sharpe)

        print(f'[FOLD {fold+1}] {sharpe=}')

    print(f'Training finished\n Mean sharpe = {np.mean(res)}')
    return np.mean(res)

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
    df = df.drop_duplicates(subset=['RowId'], keep='first')
    df = df.drop('Date_y', axis=1).rename(columns={'Date_x': 'Date'})
    df = df.drop([col for col in df.columns if df[col].dtype=='object'], axis=1)  # datetime cols except `Date`
    df = df.replace([np.inf, -np.inf], 0)
    df = df.dropna(subset='Target', how='any')

    y = df.Target
    X = df.drop(['SecuritiesCode', 'Target', 'CurrentPeriodEndDate'], axis=1)

    groups = pd.factorize(
        X['Date'].dt.day.astype(str) + 
        '_' + X['Date'].dt.month.astype(str) + 
        '_' + X['Date'].dt.year.astype(str))[0]

    cv = PurgedGroupTimeSeriesSplit(
        n_splits = FOLDS, 
        group_gap = GROUP_GAP, 
        max_train_group_size = MAX_TRAIN_GROUP_SIZE, 
        max_test_group_size = MAX_TEST_GROUP_SIZE)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "jpx-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name)
    study.optimize(objective_wrapper(cv, X, y, groups), n_trials=100)

if __name__ == '__main__':
    main()