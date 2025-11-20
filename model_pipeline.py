from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline

FEATURES = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 
            'PT08.S5(O3)', 'T', 'RH', 'AH']
TARGET = ['NMHC(GT)', 'CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

def get_train_test() -> pd.DataFrame:
    air_quality = fetch_ucirepo(id=360)
    origin = air_quality.data.features
    # process data
    origin.replace(-200, np.nan, inplace=True)
    origin['Datetime'] = origin['Date'] + ' ' + origin['Time']
    origin.drop(columns=['Date', 'Time'], inplace=True)
    origin['Datetime'] = pd.to_datetime(origin['Datetime'])
    origin.set_index('Datetime', inplace=True)
    origin = origin.sort_index(axis=1)
    origin=origin[['NMHC(GT)', 'CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)',
               'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
               'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']]
    train_dataset = origin[:'2004-12-31 23:00:00']
    test_dataset = origin['2005-01-01 00:00:00':]
    return train_dataset, test_dataset

class MissingValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=5)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame=None):
        self.scaler.fit(X)
        X_scaled = pd.DataFrame(self.scaler.transform(X), 
                                columns=X.columns, 
                                index=X.index)
        self.imputer.fit(X_scaled)
        return self

    def transform(self, X: pd.DataFrame):
        X_scaled = pd.DataFrame(self.scaler.transform(X), 
                                columns=X.columns, 
                                index=X.index)
        X_imputed = pd.DataFrame(self.scaler.inverse_transform(self.imputer.transform(X_scaled)), 
                                 columns=X.columns, 
                                 index=X.index)
        return X_imputed
    
class CreateFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lags: list, window_sizes: list):
        self.lags_ = lags
        self.window_sizes_ = window_sizes

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame=None):
        self.columns_ = X.columns
        return self
    

    def create_lagged_features(self, X:pd.DataFrame) -> pd.DataFrame:
        lagged_data = pd.DataFrame(index=X.index)
        lagged_features = {}
        for col in self.columns_:
            for lag in self.lags_:
                lagged_col_name = f"{col}_lag{lag}"
                lagged_features[lagged_col_name] = X[col].shift(lag)
        lagged_data = pd.concat([pd.DataFrame(lagged_features)], axis=1)
        return lagged_data

    def create_rolling_features(self, X:pd.DataFrame) -> pd.DataFrame:
        rolling_data = pd.DataFrame(index=X.index)
        for col in self.columns_:
            for window in self.window_sizes_:
                rolling_col_name = f"{col}_roll{window}"
                rolling_data[rolling_col_name] = X[col].rolling(window=window, min_periods=1).mean()
        return rolling_data

    def create_cyc_features(self, X:pd.DataFrame) -> pd.DataFrame:
        cyc_data = pd.DataFrame(index=X.index)
        cyc_data['Hour_sin'] = np.sin(2 * np.pi * X.index.hour / 24)
        cyc_data['Hour_cos'] = np.cos(2 * np.pi * X.index.hour / 24)
        cyc_data['DayOfWeek_sin'] = np.sin(2 * np.pi * X.index.dayofweek / 7)
        cyc_data['DayOfWeek_cos'] = np.cos(2 * np.pi * X.index.dayofweek / 7)
        cyc_data['Month_sin'] = np.sin(2 * np.pi * (X.index.month - 1) / 12)
        cyc_data['Month_cos'] = np.cos(2 * np.pi * (X.index.month - 1) / 12)
        return cyc_data

    def transform(self, X: pd.DataFrame):
        lagged_features = self.create_lagged_features(X)
        rolling_features = self.create_rolling_features(X)
        cyc_features = self.create_cyc_features(X)
        combined_features = pd.concat([X, lagged_features, rolling_features, cyc_features], axis=1)
        return combined_features
    
def select_non_nan_rows(X, Y):
    mask = Y.notna().all(axis=1) & X.notna().all(axis=1)
    return X[mask], Y[mask]

def preprocess_data(train_dataset, test_dataset, target_col):
    col = target_col

    X_train, Y_train = train_dataset[FEATURES], train_dataset[[col]]  # Drop NaN in target for training set
    X_test, Y_test = test_dataset[FEATURES], test_dataset[[col]]  # Drop NaN in target for test set

    data_preprocessing_pipeline = Pipeline(steps=[
        ('missing', MissingValueTransformer()),
        ('features', CreateFeaturesTransformer(lags=[1,3,6,12,24], window_sizes=[3,6,12])),
    ])

    X_train = data_preprocessing_pipeline.fit_transform(X_train)
    X_test = data_preprocessing_pipeline.transform(X_test)
    X_train, Y_train = select_non_nan_rows(X_train, Y_train)
    X_test, Y_test = select_non_nan_rows(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test