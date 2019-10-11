import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin

from category_encoders.target_encoder import TargetEncoder

class CounterEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,cols):
        self.cols = cols
        self.encoders = {}

    def fit(self, x):
        for col in self.cols:
            self.encoders[col] = x.groupby(col).size()

    def transform(self, x):
        new_x = x[self.cols].copy()
        new_cols = []
        for col in self.cols:
            new_x['encoded_Counter_' + col] = new_x[col].map(self.encoders[col])
            new_cols.append('encoded_Counter_' + col)
        return new_x[new_cols]

class RepeatedEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder with validation within
    """
    def __init__(self, n_folds=10, n_repeats=3, random_state=0, encoder_name=None, cols=None, smoothing = None, **kwargs):
        """
        :param cols: Categorical columns
        :param encoder_name: Name of encoder
        """
        self.cols = cols
        self.kwargs = kwargs
        self.smoothing = smoothing
        self.encoder_name = encoder_name

        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.model_validation = RepeatedStratifiedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=random_state)
        self.encoder_name = encoder_name

        self.encoders_list = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        cols_representation = None
        x = x[self.cols]

        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(x, y)):
            X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            encoder = self._get_single_encoder(self.encoder_name, self.cols, self.smoothing)
            encoder.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            val_t = encoder.transform(X_val)
            self.encoders_list.append(encoder)

            if cols_representation is None:
                cols_representation = np.zeros((x.shape[0], val_t.shape[1]))

            cols_representation[val_idx, :] += val_t.values / self.n_repeats

        cols_representation = pd.DataFrame(cols_representation, columns=val_t.columns, index=x.index)
        cols_representation.columns = [f"encoded_{self.encoder_name}_{i}" for i in cols_representation.columns]

        return cols_representation

    def transform(self, x: pd.DataFrame, y: pd.Series=None) -> pd.DataFrame:
        cols_representation = None
        x = x[self.cols]

        for n_fold, encoder in enumerate(self.encoders_list):
            test_tr = encoder.transform(x)
            if cols_representation is None:
                cols_representation = np.zeros((x.shape[0], test_tr.shape[1]))
            cols_representation += test_tr.values / self.n_folds / self.n_repeats

        cols_representation = pd.DataFrame(cols_representation, columns=test_tr.columns, index=x.index)
        cols_representation.columns = [f"encoded_{self.encoder_name}_{i}" for i in cols_representation.columns]
        return cols_representation

    @staticmethod
    def _get_single_encoder(encoder_name: str, cols, smoothing):
        """
        Get encoder by its name
        :param encoder_name: Name of desired encoder
        :param cat_cols: Cat columns for encoding
        :return: Categorical encoder
        """
        if encoder_name == "TargetEncoder":
            encoder = TargetEncoder(cols=cols, smoothing=smoothing) #cols, smoothing
        else:
            raise ValueError('NO ENCODER FOUND')
        return encoder