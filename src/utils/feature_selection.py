from abc import ABC, abstractmethod
from typing import Tuple

import shap
import numpy as np
import pandas as pd
import optuna

import pyspark
import pyspark.sql.functions as F

from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from .optimization import metric_stability


class BaseFeatureSelector(ABC):
    def __init__(self, n_folds=5, random_state=None, fit_params=True):
        self.n_folds = n_folds
        self.importances = None
        self.random_state = random_state
        self.lgm_params = None
        self.fit_params = fit_params

    def fit_lgm_params(self, x_train: np.ndarray, y_train: np.ndarray,
                      x_val: np.ndarray, y_val: np.ndarray):
        print('Fitting params')
        def objective(trial):
            params = {
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'max_depth': trial.suggest_int('max_depth', 2, 16),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'verbose': -100,
                'random_state': self.random_state
            }

            model = LGBMClassifier(**params)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), callbacks=[early_stopping(stopping_rounds=30)])

            preds_train = model.predict_proba(x_train)[:, 1]
            preds_val = model.predict_proba(x_val)[:, 1]

            metric_train = roc_auc_score(y_train, preds_train)
            metric_val = roc_auc_score(y_val, preds_val)

            metric = metric_stability(metric_train, metric_val, 0.2)

            print(f'Trial #{trial.number}. Metric: {metric}. Params: {trial.params}')
            return metric

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True)
        )

        study.optimize(objective, n_trials=30, n_jobs=1)
        params = study.best_params
        params['verbose'] = -100
        params['random_state'] = self.random_state

        return params

    def fit(self, x: np.ndarray, y: np.ndarray, columns: list, skf=None, **importances_kwargs):
        if skf is None:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
        else:
            self.n_folds = skf.n_splits

        self.importances = pd.DataFrame({'feature': columns})

        for i, (train_index, val_index) in enumerate(skf.split(x, y)):
            x_train, y_train = x[train_index], y[train_index]
            x_val, y_val = x[val_index], y[val_index]

            if self.fit_params:
                if not self.lgm_params:
                    self.lgm_params = self.fit_lgm_params(x_train, y_train, x_val, y_val)

            else:
                self.lgm_params = {
                    'max_depth': 5,
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'verbose': -100,
                    'random_state': self.random_state
                }

            model = LGBMClassifier(**self.lgm_params)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), callbacks=[early_stopping(stopping_rounds=30)])

            imp = self._get_importances_from_model(model, x_val, y_val, **importances_kwargs)

            self.importances[f'importance_{i}'] = imp

    def get_selected_features(self, threshold):
        assert self.importances is not None, 'Сначала нужно обучить, вызвав метод fit'

        # сделаем отдельно для 0 итерации
        imps = self.importances.loc[:, ['feature', 'importance_0']].sort_values('importance_0', ascending=False)  # выберем важности признаков с 0 итерации
        imps['importance_0'] /= imps['importance_0'].sum()
        imps['cumsum'] = imps['importance_0'].cumsum()  # так как мы их отнормировали, может посчитать кумулятивную сумму
        features = imps.loc[imps['cumsum'] <= threshold, 'feature'].tolist()  # возьмем только те признаки, которые по кумулятивной сумме удовлетворяют
        
        best_features = set(features)  # сделаем множество
        for i in range(1, self.n_folds):
            imps = self.importances.loc[:, ['feature', f'importance_{i}']].sort_values(f'importance_{i}', ascending=False)
            imps[f'importance_{i}'] /= imps[f'importance_{i}'].sum()
            imps['cumsum'] = imps[f'importance_{i}'].cumsum()
            features = imps.loc[imps['cumsum'] <= threshold, 'feature'].tolist()

            best_features &= set(features)  # смотрим на пересечения множеств на разных итерациях кросс-валидации

        return list(best_features)
        
    @abstractmethod
    def _get_importances_from_model(self, model, x, y, **kwargs):
        pass
    
    
class LGMFeatureSelection(BaseFeatureSelector):
    def __init__(self, n_folds=5, random_state=None, fit_params=True):
        super().__init__(n_folds)

    def _get_importances_from_model(self, model, x, y, importance_type='split'):
        return model.booster_.feature_importance(importance_type=importance_type)
    

class ShapFeatureSelection(BaseFeatureSelector):
    def __init__(self, n_folds=5, random_state=None, fit_params=True):
        super().__init__(n_folds)

    def _get_importances_from_model(self, model, x, y, 
                                    is_multiclass=False, feature_perturbation='tree_path_dependent'):
        explainer = shap.TreeExplainer(model, feature_perturbation=feature_perturbation)
        shap_values = explainer.shap_values(x)  # for each class, for each instance

        if is_multiclass:
            if isinstance(shap_values, list):
                # if shap_values in list of n_classes np.arrays of shape [n_samples, n_features]
                importances = []
                for cls_ in shap_values:
                    cls_value = np.abs(cls_).mean(axis=0)
                    importances.append(cls_value.reshape(1, -1))
                    
                importances = np.concatenate(importances).mean(axis=0)
            else:
                # if shap_values in np.array of shape [n_samples, n_features, n_classes]
                importances = np.abs(shap_values).mean(axis=(0, 2))
        else:
            if isinstance(shap_values, list):
                # if shap_values in list of n_classes np.arrays of shape [n_samples, n_features]
                importances = np.abs(shap_values[1]).mean(axis=0)
            else:
                # if shap_values in np.array of shape [n_samples, n_features]
                importances = np.abs(shap_values).mean(axis=0)
        return importances


def remove_null_cols(df: pyspark.sql.session.SparkSession, thr: float=0.95) -> Tuple[pyspark.sql.session.SparkSession, dict]:
    """
    Keep columns containig less than 'thr' percent of Nulls
    """
    df_len = df.count()
    null_stats = df.select([(F.count(F.when(F.isnull(c), c)) / df_len).alias(c) for c in df.columns]).collect()
    null_stats = [row.asDict() for row in null_stats][0]
    columns = [col for col, per_null in null_stats.items() if per_null < thr]
    return df.select(columns), null_stats


def find_correlated_columns(subset: pd.DataFrame, thr: float=0.95) -> Tuple[set, dict]:
    """
    Find columns that have correlation more than a thr and return the set of the second correlated columns
    """
    corrs = subset.corr(numeric_only=True)
    columns = corrs.columns
    n = len(columns)

    correlated_columns = []

    for i in range(n):
        if columns[i] in correlated_columns:
            continue
        for j in range(i+1, n):
            val = corrs.iloc[i, j]

            if abs(val) > thr:
                correlated_columns.append(columns[j])
    return set(correlated_columns), corrs
