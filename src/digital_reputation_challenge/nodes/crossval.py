import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from hyperopt import STATUS_OK
from digital_reputation_challenge.nodes.datatransform import get_fold_data

from kedro.config import ConfigLoader

conf_paths = ['conf/base', 'conf/local']
conf_loader = ConfigLoader(conf_paths)
conf_parameters = conf_loader.get('parameters*', 'parameters*/**')

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold


class CV_score:
    def __init__(self, params, cols_all, col_target, cols_cat='auto', num_boost_round=99999,
                 early_stopping_rounds=50, valid=True):
        self.params = params
        self.cols_all = cols_all
        self.col_target = col_target
        self.cols_cat = cols_cat
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.valid = valid
        self.models = {}
        self.scores = None
        self.score_max = None
        self.num_boost_optimal = None
        self.std = None

    def fit(self, df, folds):
        log = logging.getLogger(__name__)
        scores = {}
        scores_avg = []
        fold_list = np.sort(folds['fold'].unique())
        log.info(self.params)

        x = df[self.cols_all]
        y = df[self.col_target]

        self.model_validation = RepeatedStratifiedKFold(n_splits=10, n_repeats=15, random_state=0)
        # for fold in fold_list:
        for fold, (train_idx, val_idx) in enumerate(self.model_validation.split(np.zeros(x.shape), y)):
            X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(data=X_train.astype(float), label=y_train,
                                 categorical_feature=self.cols_cat)
            dvalid = lgb.Dataset(data=X_val.astype(float), label=y_val,
                                 categorical_feature=self.cols_cat)

            log.info(f'CROSSVALIDATION FOLD {fold} START')
            #
            # train, test = get_fold_data(df,folds,fold,fold_list, col_target = self.col_target)
            #
            # # Подготовка данных в нужном формате
            # dtrain = lgb.Dataset(data=train[self.cols_all].astype(float), label=train[self.col_target],
            #                      categorical_feature=self.cols_cat)
            # dvalid = lgb.Dataset(data=test[self.cols_all].astype(float), label=test[self.col_target],
            #                      categorical_feature=self.cols_cat)

            # Обучение
            evals_result = {}
            if self.valid:
                model = lgb.train(params=self.params,
                                  train_set=dtrain,
                                  valid_sets=[dtrain, dvalid],
                                  valid_names=['train', 'eval'],
                                  num_boost_round=self.num_boost_round,
                                  evals_result=evals_result,
                                  categorical_feature=self.cols_cat,
                                  early_stopping_rounds=self.early_stopping_rounds,
                                  verbose_eval=False)
            else:
                model = lgb.train(params=self.params,
                                  train_set=dtrain,
                                  num_boost_round=self.num_boost_round,
                                  categorical_feature=self.cols_cat,
                                  verbose_eval=False)

            self.models[fold] = model
            if self.valid:
                # Построение прогнозов при разном виде взаимодействия
                scores[fold] = evals_result['eval']['auc']
                scores_avg.append(np.max(evals_result['eval']['auc']))

        if self.valid:
            self.scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in scores.items()]))
            mask = self.scores.isnull().sum(axis=1) == 0
            self.num_boost_optimal = np.argmax(self.scores[mask].mean(axis=1))
            # self.score_max = self.scores[mask].mean(axis=1)[self.num_boost_optimal]
            self.score_max = np.mean(scores_avg)
            # self.std = self.scores[mask].std(axis=1)[self.num_boost_optimal]
            self.std = np.std(scores_avg)

            # self.score_max_lastfold = self.scores[max(fold_list)].loc[self.num_boost_optimal]
            # self.score_max_withoutlastfold = self.scores.iloc[:, range(0, int(max(fold_list)))].mean(axis=1).loc[
            #     self.num_boost_optimal]
            result = {'loss': -self.score_max,
                      'status': STATUS_OK,
                      'std': self.std,
                      'score_max': self.score_max,
                      'scores_all': scores_avg
                      # 'num_boost': int(self.num_boost_optimal),
                      # 'score_max_lastfold': self.score_max_lastfold,
                      # 'score_max_withoutlastfold': self.score_max_withoutlastfold,
                      }
            log.info(result)
            return result
        return self

    def transform_train(self, df, folds):
        # fold_list = np.sort(folds['fold'].unique())
        # for fold in fold_list:
        #     train, test = get_fold_data(df,folds,fold,fold_list)

        x = df[self.cols_all]
        y = df[self.col_target]

        for fold, (train_idx, val_idx) in enumerate(self.model_validation.split(np.zeros(x.shape), y)):
            X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Подготовка данных в нужном формате
            model = self.models[fold]
            #
            # df.loc[test.index, 'PREDICT'] = \
            #     model.predict(test[model.feature_name()].astype(float),
            #                   num_iteration = self.num_boost_optimal)
            df.loc[X_val.index, 'PREDICT'] = \
                model.predict(X_val[model.feature_name()].astype(float),) / 10
                              # num_iteration=self.num_boost_optimal) / 10

        return df['PREDICT']

    def transform_test(self, test):
        models_len = len(self.models.keys())

        test['PREDICT'] = 0
        for fold in self.models.keys():
            model = self.models[fold]
            test['PREDICT'] += model.predict(test[model.feature_name()].astype(float),) / models_len
                                             # num_iteration=self.num_boost_optimal) / models_len

        return test['PREDICT']

    def shap(self, df: pd.DataFrame, folds):
        fig = plt.figure(figsize=(10, 10))
        col_id = conf_parameters['df_app']['col_id']
        log = logging.getLogger(__name__)
        shap_df_fin = pd.DataFrame(columns=['feature'])

        #
        # fold_list = np.sort(folds['fold'].unique())
        # for fold in fold_list:
        #     train, test = get_fold_data(df,folds,fold,fold_list)
        # Подготовка данных в нужном формате


        x = df[self.cols_all]
        y = df[self.col_target]

        for fold, (train_idx, val_idx) in enumerate(self.model_validation.split(np.zeros(x.shape), y)):
            X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self.models[fold]
            explainer = shap.TreeExplainer(model)
            df_sample = X_val[model.feature_name()].sample(n=500, random_state=0, replace=True).astype(float)
            shap_values = explainer.shap_values(df_sample)[0]
            shap_df = pd.DataFrame(zip(model.feature_name(), np.mean(np.abs(shap_values), axis=0)),
                                   columns=['feature', 'shap_' + str(fold)])
            shap_df_fin = pd.merge(shap_df_fin, shap_df, how='outer', on='feature')

        shap_feature_stats = shap_df_fin.set_index('feature').agg(['mean', 'std'], axis=1).sort_values('mean',
                                                                                                       ascending=False)
        cols_best = shap_feature_stats[:30].index

        best_features = shap_df_fin.loc[shap_df_fin['feature'].isin(cols_best)]
        best_features_melt = pd.melt(best_features, id_vars=['feature'],
                                     value_vars=[feature for feature in best_features.columns.values.tolist() if
                                                 feature not in ['feature']])

        sns.barplot(x='value', y='feature', data=best_features_melt, estimator=np.mean, order=cols_best)
        return fig, shap_feature_stats

    def shap_summary_plot(self, df: pd.DataFrame, folds):
        fig = plt.figure()
        col_id = conf_parameters['df_app']['col_id']
        log = logging.getLogger(__name__)
        shap_df_fin = pd.DataFrame(columns=['feature'])

        fold_list = np.sort(folds['fold'].unique())
        for fold in fold_list[:1]:
            train, test = get_fold_data(df, folds, fold, fold_list)
            # Подготовка данных в нужном формате
            model = self.models[fold]
            explainer = shap.TreeExplainer(model)
            df_sample = test[model.feature_name()].sample(n=500, random_state=0, replace=True).astype(float)
            shap_values = explainer.shap_values(df_sample)[0]
            shap_df = pd.DataFrame(zip(model.feature_name(), np.mean(np.abs(shap_values), axis=0)),
                                   columns=['feature', 'shap_' + str(fold)])
            shap_df_fin = pd.merge(shap_df_fin, shap_df, how='outer', on='feature')

        shap.summary_plot(shap_values, df_sample, show=False, )
        return fig