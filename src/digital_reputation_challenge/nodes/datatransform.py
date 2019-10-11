import pandas as pd
import numpy as np
import logging
from kedro.config import ConfigLoader
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold
from digital_reputation_challenge.nodes.target_encoding import RepeatedEncoder


conf_paths = ['conf/base', 'conf/local']
conf_loader = ConfigLoader(conf_paths)
conf_parameters = conf_loader.get('parameters*', 'parameters*/**')

col_id = conf_parameters['df_app']['col_id']
col_target = conf_parameters['df_app']['col_target']
cols_delete = conf_parameters['cols_delete']
col_dt = ''
folds = conf_parameters['df_app']['folds']
oot = conf_parameters['df_app']['oot']
cv_byclient = conf_parameters['df_app']['cv_byclient']

n_folds = conf_parameters['targetencoding']['n_folds']
n_repeats = conf_parameters['targetencoding']['n_repeats']
smoothing = conf_parameters['targetencoding']['smoothing']
random_state = conf_parameters['targetencoding']['random_state']
cols_cat = conf_parameters['cols_cat']

def fit_folds(df_app, folds=folds, col_dt='', col_id = col_id, oot=oot, cv_byclient=False, to_delete = False,
              stratified=False):
    log = logging.getLogger(__name__)

    df_app['fold']=np.nan

    if to_delete:
        # удаление данные где не проставлен target
        mask_target = ~df_app[col_target].isnull()
        df_app = df_app[mask_target].copy()

    if oot:
        # отложенная выборка во времени
        target_ind = df_app.sort_values(col_dt)[-int(df_app.shape[0] * (1/folds)):].index
        df_app.loc[df_app.index.isin(target_ind), 'fold'] = folds-1
        folds=folds-1


    mask_fold = df_app['fold'].isnull()
    X = df_app[mask_fold].copy()

    if cv_byclient:
        # кросс валидация по клиентам
        unique_clients = X['borrower_id'].unique()
        kf = KFold(n_splits = folds, shuffle = True, random_state=0)
        kf.get_n_splits(unique_clients)
        for i, (train_index, test_index) in enumerate(kf.split(unique_clients)):
            test_mask = X['borrower_id'].isin(unique_clients[test_index])
            df_app.loc[df_app[col_id].isin(X.loc[test_mask, col_id]), 'fold'] = i
    elif stratified:
        # обычная кросс валидация
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        skf.get_n_splits(np.zeros(X.shape), X[col_target])
        for i, (train_index, test_index) in enumerate(skf.split(np.zeros(X.shape), X[col_target])):
            test_mask = X.iloc[test_index].index
            df_app.loc[df_app[col_id].isin(X.loc[test_mask, col_id]), 'fold'] = i
    else:
        # обычная кросс валидация
        skf = KFold(n_splits=folds, shuffle=True, random_state=0)
        skf.get_n_splits(np.zeros(X.shape))
        for i, (train_index, test_index) in enumerate(skf.split(np.zeros(X.shape))):
            test_mask = X.iloc[test_index].index
            df_app.loc[df_app[col_id].isin(X.loc[test_mask, col_id]), 'fold'] = i

    if col_dt!='':
        stats = df_app.groupby('fold')[col_dt].agg(['min','max','count'])
        log.info('SPLIT STATISTICS:')
        log.info(stats)
    return df_app[[col_id, 'fold']]



def get_fold_data(df: pd.DataFrame, folds: pd.DataFrame, fold, fold_list, col_target=None):
    test_transactions = folds[folds['fold'] == fold][col_id]
    train_transactions = folds[(folds['fold'] != fold) & (folds['fold'] != max(fold_list))][col_id]

    test = df.loc[test_transactions]
    train = df.loc[train_transactions]


    # repeated_encoder = RepeatedEncoder(n_folds=n_folds,n_repeats=n_repeats,random_state=random_state,
    #                                    encoder_name="TargetEncoder",cols=cols_cat,smoothing=smoothing)
    #
    # train_t = repeated_encoder.fit(train, train[col_target])
    # test_t = repeated_encoder.transform(test)
    #
    # return pd.concat([train,train_t],axis=1), pd.concat([test,test_t],axis=1)

    return train, test

def get_cols(df, col_target=None):
    cols_X = [col for col in df.columns if col[0]=='X']
    cols_Y = [col for col in df.columns if col[0]=='Y']
    if col_target:
        cols_X = [col for col in cols_X if col not in cols_delete[col_target]]
    return cols_X, cols_Y