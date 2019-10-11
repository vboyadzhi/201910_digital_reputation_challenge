
import pandas as pd
import numpy as np
import logging
from lightfm import LightFM
from lightfm.data import Dataset
from typing import Dict, List
from digital_reputation_challenge.nodes.datatransform import fit_folds, get_cols
from digital_reputation_challenge.nodes.crossval import CV_score
from digital_reputation_challenge.nodes.target_encoding import RepeatedEncoder, CounterEncoder
from sklearn.metrics import roc_auc_score
from digital_reputation_challenge.nodes.WOE_V018 import WOE

from sklearn.cluster import KMeans
import scipy
from digital_reputation_challenge.io.matplotlib_io import MatplotlibWriter
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from hyperopt.pyll import scope
from itertools import combinations
from kedro.config import ConfigLoader
conf_paths = ['conf/base', 'conf/local']
conf_loader = ConfigLoader(conf_paths)
conf_parameters = conf_loader.get('parameters*', 'parameters*/**')
conf_catalog = conf_loader.get('catalog*', 'catalog*/**')


col_target = conf_parameters['df_app']['col_target']
col_id = conf_parameters['df_app']['col_id']
# col_dt = conf_parameters['df_app']['col_dt']
params_lgb = conf_parameters['lightgbm']['params']
max_evals = conf_parameters['hyperopt']['max_evals']
random_state = conf_parameters['random_state']

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold
import umap

def umap_node(X2_train, X2_test, Y_train):
    def dummy_fun(doc):
        return doc

    tfidf = TfidfVectorizer(
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        binary=True,
    )

    X2 = pd.concat([X2_train, X2_test])
    X2 = X2.groupby('id').apply(lambda x: x['A'].astype(str).values.tolist())
    X2_tfidf = tfidf.fit_transform(X2)
    nb = umap.UMAP(n_components=40,random_state=0)

    Y_train = Y_train.set_index('id')
    Y_train.columns = ['Y_' + i for i in Y_train.columns]

    result = pd.DataFrame(nb.fit_transform(X2_tfidf),index=X2.index)
    result.columns = ['X2_umap_' + str(i) for i in result.columns]
    return result

def svd_node(X2_train, X2_test, Y_train):
    def dummy_fun(doc):
        return doc

    tfidf = TfidfVectorizer(
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        binary=True,
    )

    X2 = pd.concat([X2_train, X2_test])
    X2 = X2.groupby('id').apply(lambda x: x['A'].astype(str).values.tolist())
    X2_tfidf = tfidf.fit_transform(X2)
    nb = TruncatedSVD(n_components=40,random_state=0)

    Y_train = Y_train.set_index('id')
    Y_train.columns = ['Y_' + i for i in Y_train.columns]

    result = pd.DataFrame(nb.fit_transform(X2_tfidf),index=X2.index)
    result.columns = ['X2_svd_' + str(i) for i in result.columns]
    return result

def lda_node(X2_train, X2_test, Y_train):
    def dummy_fun(doc):
        return doc

    tfidf = TfidfVectorizer(
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        binary=True,
    )

    X2 = pd.concat([X2_train, X2_test])
    X2 = X2.groupby('id').apply(lambda x: x['A'].astype(str).values.tolist())
    X2_tfidf = tfidf.fit_transform(X2)
    nb = LatentDirichletAllocation()

    Y_train = Y_train.set_index('id')
    Y_train.columns = ['Y_' + i for i in Y_train.columns]

    result = pd.DataFrame(nb.fit_transform(X2_tfidf),index=X2.index)
    result.columns = ['X2_lda_' + str(i) for i in result.columns]
    return result

def naive_node(X2_train, X2_test, Y_train):
    def dummy_fun(doc):
        return doc

    tfidf = TfidfVectorizer(
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        binary=True,
    )

    X2 = pd.concat([X2_train, X2_test])
    X2 = X2.groupby('id').apply(lambda x: x['A'].astype(str).values.tolist())
    X2_tfidf = tfidf.fit_transform(X2)

    Y_train = Y_train.set_index('id')
    Y_train.columns = ['Y_' + i for i in Y_train.columns]
    test_index = X2.index[~X2.index.isin(Y_train.index)]
    result = pd.DataFrame(index=X2.index)
    for col in Y_train.columns:
        result['X2_nb_predict_' + col] = 0

    alpha = {'Y_1': {'alpha': 0.13315028794354022},
             'Y_2': {'alpha': 0.8452951463886212},
             'Y_3': {'alpha': 0.14592826346371515},
             'Y_4': {'alpha': 0.10164552441679695},
             'Y_5': {'alpha': 0.6753417707282328}}
    for col in Y_train.columns:
        model_validation = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)

        nb = MultinomialNB(alpha=alpha[col]['alpha'])
        for fold, (train_idx, val_idx) in enumerate(model_validation.split(np.zeros(Y_train[[col]].shape), Y_train[col])):

            nb.fit(X2_tfidf[Y_train.iloc[train_idx].index], Y_train.iloc[train_idx][col])
            result.loc[Y_train.iloc[val_idx].index, 'X2_nb_predict_'+col] += nb.predict_proba(X2_tfidf[Y_train.iloc[val_idx].index])[:, 1]/10
            result.loc[test_index,'X2_nb_predict_'+col] += nb.predict_proba(X2_tfidf[test_index])[:, 1]/50
    return result

def kmean_node(X2_train,X2_test):
    X2 = pd.concat([X2_train, X2_test])
    del X2_train, X2_test

    X2 = X2.groupby(['id', 'A']).size().astype(np.int8)
    X2 = X2.reset_index()
    X2.columns = ['id', 'A', 'val']

    sparse_mat = scipy.sparse.coo_matrix((X2.val, (X2.id, X2.A)))

    def pairwise_jaccard(X):
        """Computes the Jaccard distance between the rows of `X`.
        """
        X = X.astype(bool).astype(int)

        intrsct = X.dot(X.T)
        row_sums = intrsct.diagonal()
        unions = row_sums[:, None] + row_sums - intrsct
        dist = intrsct / unions
        return dist

    similarities = pairwise_jaccard(sparse_mat)
    similarities = pd.DataFrame(similarities)
    km = KMeans()
    km.fit(similarities)

    a = km.transform(similarities)
    a = pd.DataFrame(a)
    a.columns = a.columns.map(lambda x: 'X2_kmean_' + str(x))
    return a

def merge_df(X1:pd.DataFrame,X2:pd.DataFrame,X3:pd.DataFrame,lightfm_embed:pd.DataFrame,kmean,naive,lda,svd,umap,Y:pd.DataFrame=None)->pd.DataFrame:
    X1 = X1.set_index('id')
    X1.columns = ['X1_' + i for i in X1.columns]
    X3 = X3.set_index('id')
    X3.columns = ['X3_' + i for i in X3.columns]
    X3.apply(lambda a: a / a[a > 0].min(), axis=1)

    for col1, col2 in combinations(X1.columns , 2):
        X1[col1+'_mul_'+col2] = X1[col1]*X1[col2]

    # for col1, col2, col3 in combinations(X1.columns, 3):
    #     X1[col1 + '_mul_' + col2 + '_mul_' + col3] = X1[col1] * X1[col2] * X1[col3]

    lightfm_embed.columns = ['X_lightfm_' + str(i) for i in lightfm_embed.columns]

    if Y is not None:
        Y = Y.set_index('id')
        Y.columns = ['Y_' + i for i in Y.columns]
        X = pd.concat([X1,  lightfm_embed, kmean, naive, lda, svd,umap, Y], join='inner', axis=1)
    else:
        X = pd.concat([X1,  lightfm_embed, kmean, naive, lda, svd,umap], join='inner', axis=1)
    #
    # np.random.RandomState(seed=42)
    # for i in range(5):
    #     X['X_rand'+str(i)] = np.random.randint(0, X.shape[0], X.shape[0])

    return X


def app_split(df_app)->pd.DataFrame:
    df_app = fit_folds(df_app)
    return df_app


def cross_validation(df_train: pd.DataFrame, folds: pd.DataFrame, best=None):
    log = logging.getLogger(__name__)

    results={}
    models={}
    avg = 0
    for col in col_target:
        cols_all, _ = get_cols(df_train, col)
        if best:
            params = best[col]
        else:
            params = params_lgb

        log.info(f"CROSS VALIDATION COLUMN {col}")
        cv_score = CV_score(params=params,
                            cols_all=cols_all,
                            col_target=col,
                            num_boost_round=999999,
                            early_stopping_rounds=50,
                            valid=True)

        results[col] = cv_score.fit(df=df_train, folds=folds)
        avg += results[col]["score_max"]/len(col_target)
        models[col] = cv_score
    results['AVG'] = avg
    return [results, models]

def cross_validation_train(df_train: pd.DataFrame, folds: pd.DataFrame, crossval_models):
    log = logging.getLogger(__name__)

    results=pd.DataFrame(index=df_train.index)
    for col in col_target:
        cols_all, _ = get_cols(df_train, col)
        cv_score = crossval_models[col]
        results[col] = cv_score.transform_train(df=df_train, folds=folds)
    return results

def cross_validation_shap(df_train: pd.DataFrame, folds: pd.DataFrame, crossval_models):
    log = logging.getLogger(__name__)

    results_shap_reg={}
    results_shap_sum={}
    shap_feature_stats={}
    for col in col_target:
        cols_all, _ = get_cols(df_train, col)
        log.info(f'SHAP {col} START')
        cv_score = crossval_models[col]
        results_shap_reg[col], shap_feature_stats[col] = cv_score.shap(df=df_train, folds=folds)
        # results_shap_sum[col] = cv_score.shap_summary_plot(df=df_train, folds=folds)


    return [results_shap_reg, shap_feature_stats]


def find_hyperopt(df_train: pd.DataFrame, folds: pd.DataFrame) -> Dict:
    log = logging.getLogger(__name__)
    cols_all, col_target = get_cols(df_train)

    results={}
    space = {
        'num_leaves': scope.int(hp.quniform('num_leaves', 3, 100, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 10, 70, 1)),
        'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 5, 150, 1)),
        'feature_fraction': hp.uniform('feature_fraction', 0.85, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.85, 1.0),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'lambda_l1': hp.uniform('lambda_l1', 1e-4, 2),
        'lambda_l2': hp.uniform('lambda_l2', 1e-4, 2),
        'seed': random_state,
        'feature_fraction_seed': random_state,
        'bagging_seed': random_state,
        'drop_seed': random_state,
        'data_random_seed': random_state,
        'verbose': -1,
        'bagging_freq': 5,
        'max_bin': 255,
        'learning_rate': 0.001,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
    }
    for col in col_target:
        cols_all, _ = get_cols(df_train, col)
        def score(params):
            cv_score = CV_score(params = params,
                                cols_all=cols_all,
                                col_target=col,
                                num_boost_round=99999999,
                                early_stopping_rounds=50,
                                valid=True)
            return cv_score.fit(df=df_train, folds=folds)

        trials = Trials()
        best = fmin(fn=score,
                    space=space,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=max_evals
                    )
        results[col] = space_eval(space, best)
    return results


def cross_validation_test(df_test: pd.DataFrame, crossval_models) -> Dict:
    log = logging.getLogger(__name__)
    cols_all, _ = get_cols(df_test)

    results=pd.DataFrame(index=df_test.index)
    for col in col_target:
        cv_score = crossval_models[col]
        results[col] = cv_score.transform_test(df_test)

    return [results]

def submit(cv_test):
    Y = cv_test.reset_index()
    Y.columns = ['id', '1', '2', '3', '4', '5']
    return Y


def target_enc(df_train: pd.DataFrame, df_test: pd.DataFrame):
    log = logging.getLogger(__name__)
    cols_all, cols_target = get_cols(df_train)

    enc = WOE()
    for col in cols_target:
        a = enc.fit_transform(df_train[cols_all], df_train[col])
        a.columns = a.columns.map(lambda x: x + '_encoding_' + col)

        b = a.T.apply(lambda x: 1 - 2 * roc_auc_score(df_train[col], x), axis=1)
        merge_cols = b[b > 0.04].index
        print(merge_cols)
        df_train = pd.concat([df_train, a[merge_cols]], axis=1)
        a = enc.transform(df_test[cols_all])
        a.columns = a.columns.map(lambda x: x + '_encoding_' + col)
        df_test = pd.concat([df_test, a[merge_cols]], axis=1)

    return [df_train, df_test]


def lightfm_node(X1_train, X2_train, X1_test, X2_test):
    X2 = pd.concat([X2_train, X2_test])
    X1 = pd.concat([X1_train, X1_test]).set_index('id')

    X1.columns = ['X1_' + i for i in X1.columns]

    X1['X1_5'] = pd.qcut(X1['X1_5'], np.arange(0, 1, 0.1), duplicates='drop')
    X1['X1_8'] = pd.qcut(X1['X1_8'], np.arange(0, 1, 0.1), duplicates='drop')
    X1['X1_6'] = pd.qcut(X1['X1_6'], np.arange(0, 1, 0.1), duplicates='drop')

    for col in ['X1_6', 'X1_8', 'X1_5', 'X1_1', 'X1_13']:
        X1[col] = X1[col].map(lambda x: '{' + col + '}_{' + str(x) + '}')

    X1 = X1.reset_index()

    from lightfm.data import Dataset
    dataset = Dataset()
    dataset.fit(users=(x for x in X2['id']),
                items=(x for x in X2['A'])
                )

    dataset.fit_partial(users=(x for x in X1['id']),
                        user_features=(x for x in X1['X1_1']))
    dataset.fit_partial(users=(x for x in X1['id']),
                        user_features=(x for x in X1['X1_13']))
    dataset.fit_partial(users = (x for x in X1['id']),
                        user_features = (x for x in X1['X1_5']))
    dataset.fit_partial(users = (x for x in X1['id']),
                        user_features = (x for x in X1['X1_8']))
    dataset.fit_partial(users = (x for x in X1['id']),
                        user_features = (x for x in X1['X1_6']))

    user_features = dataset.build_user_features([(x[1]['id'],
                                                  x[1][['X1_1', 'X1_13', 'X1_5', 'X1_8', 'X1_6']].values.tolist())
                                                 for x in X1.iterrows()],
                                                normalize=True)

    (interactions, weights) = dataset.build_interactions(zip(*X2[['id', 'A']].values.T))

    model = LightFM(no_components=32, learning_rate=0.04, loss='bpr', max_sampled=55, random_state=0)
    num_epochs = 20
    for i in range(num_epochs):
        model.fit_partial(
            interactions,
            user_features=user_features
        )

    users_mapping, user_features_mapping, assets_mapping, asset_features_mapping = dataset.mapping()
    user_features_mapping_inv = {j: i for i, j in user_features_mapping.items()}

    tag_embeddings = (model.user_embeddings.T / np.linalg.norm(model.user_embeddings, axis=1)).T

    lightfm_embed = pd.DataFrame(tag_embeddings[:len(users_mapping)], index=X1['id'])

    return lightfm_embed