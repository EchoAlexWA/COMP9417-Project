from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from torch.backends.cudnn import enabled
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.model_selection import PredefinedSplit

# region constants
def __constants__():
    pass
FEATURES = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 
            'PT08.S5(O3)', 'T', 'RH', 'AH']
TARGET = ['NMHC(GT)', 'CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
VALIDATION_PORTION = 0.2
# endregion

# region data processing functions
def __data_processing_functions__():
    pass
def basic_preprocess():
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
    return origin

def get_train_test(path:str='origin_data/air_quality_raw.csv' ,if_NMHC: bool=False) -> pd.DataFrame:
    origin = pd.read_csv(
        path,
        parse_dates=["Datetime"],  # 指定第一列转换为 datetime
        index_col="Datetime",  # 设置为 index
    )

    if not if_NMHC:
        origin['Year'] = origin.index.year
        train_dataset = origin[origin['Year'] < 2005]
        test_dataset = origin[origin['Year'] >= 2005]
        train_dataset = train_dataset.drop(columns=['Year'])
        test_dataset = test_dataset.drop(columns=['Year'])
    else:
        no_nan_max = origin['NMHC(GT)'].dropna().index.max()
        origin = origin.loc[:no_nan_max]
        train_dataset = origin.iloc[:int(len(origin) * 0.8)]
        test_dataset = origin.iloc[int(len(origin) * 0.8):]
    return train_dataset, test_dataset

def select_non_nan_rows(X, Y):
    mask = Y.notna().all(axis=1) & X.notna().all(axis=1)
    return X[mask], Y[mask]

def preprocess_data(train_dataset, test_dataset, target_col, params: dict):
    col = target_col
    lags = params.get('lags', [1,3,6,12,24])
    window_sizes = params.get('window_sizes', [3,6,12])
    enabled = params.get('anomaly_detection_enabled', False)

    X_train, Y_train = train_dataset[FEATURES], train_dataset[[col]]
    Y_train = pd.concat([Y_train.shift(-i) for i in [1,6,12,24]], axis=1)
    Y_train.columns = [Y_train.columns[0] + f"_t+{i}" for i in [1,6,12,24]]
    X_test, Y_test = test_dataset[FEATURES], test_dataset[[col]]
    Y_test = pd.concat([Y_test.shift(-i) for i in [0,1,6,12,24]], axis=1)
    Y_test.columns = [Y_test.columns[0] + f"_t+{i}" for i in [0,1,6,12,24]]

    data_preprocessing_pipeline = Pipeline(steps=[
        ('missing', MissingValueTransformer()),
        ('anomaly', IsolationForestFilter(contamination=0.01, enabled=enabled)),
        ('features', CreateFeaturesTransformer(lags=lags, window_sizes=window_sizes)),
    ])

    X_train = data_preprocessing_pipeline.fit_transform(X_train)
    data_preprocessing_pipeline.set_params(anomaly__enabled=False)
    X_test = data_preprocessing_pipeline.transform(X_test)
    X_train, Y_train = select_non_nan_rows(X_train, Y_train)
    X_test, Y_test = select_non_nan_rows(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def data_formatted(train_dataset, test_dataset, target_column, params: dict, reg: bool=True):
    '''
    Prepare data for model training and evaluation
    '''
    if reg:
        X_train, Y_train, X_test, Y_test = preprocess_data(train_dataset, test_dataset, target_column, params)
        X_train = X_train.to_numpy()
        Y_train = Y_train.to_numpy()
        scaler_Y = MinMaxScaler()
        Y_train = scaler_Y.fit_transform(Y_train)
        X_test = X_test.to_numpy()
        Y_baseline = Y_test.to_numpy()[:, 0]
        Y_test = Y_test.to_numpy()[:, 1:]
        Y_baseline = Y_baseline.reshape(-1, 1).repeat(axis=1, repeats=Y_test.shape[1])
        Y_test = scaler_Y.transform(Y_test)
        return X_train, Y_train, X_test, Y_test, Y_baseline, scaler_Y
    else:
        X_train, Y_train, X_test, Y_test = preprocess_data(train_dataset, test_dataset, target_column, params)
        X_train = X_train.to_numpy()
        for col in Y_train.columns:
            Y_train[col+'class'] = pd.cut(
                Y_train[col],
                bins=[-float('inf'), 1.5, 2.5, float('inf')],  # 边界
                labels=['low', 'mid', 'high'],                # 类别名称
                right=False                                   # 区间左闭右开 [ )
            ).cat.codes
        Y_train.drop(columns=Y_train.columns[:len(Y_train.columns)//2], inplace=True)
        Y_train = Y_train.to_numpy()

        X_test = X_test.to_numpy()
        for col in Y_test.columns:
            Y_test[col+'class'] = pd.cut(
                Y_test[col],
                bins=[-float('inf'), 1.5, 2.5, float('inf')],  # 边界
                labels=['low', 'mid', 'high'],                # 类别名称
                right=False                                   # 区间左闭右开 [ )
            ).cat.codes
        Y_test.drop(columns=Y_test.columns[:len(Y_test.columns)//2], inplace=True)
        Y_test = Y_test.to_numpy()
        Y_baseline = Y_test[:, 0]
        Y_test = Y_test[:, 1:]
        Y_baseline = Y_baseline.reshape(-1, 1).repeat(axis=1, repeats=Y_test.shape[1])
        return X_train, Y_train, X_test, Y_test, Y_baseline, None
# endregion

# region custom transformers
def __custom_transformers__():
    pass
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

    def transform(self, X: pd.DataFrame, Y: pd.DataFrame=None):
        X_scaled = pd.DataFrame(self.scaler.transform(X), 
                                columns=X.columns, 
                                index=X.index)
        X_imputed = pd.DataFrame(self.scaler.inverse_transform(self.imputer.transform(X_scaled)), 
                                 columns=X.columns, 
                                 index=X.index)
        return X_imputed

class IsolationForestFilter(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.01, random_state=42, enabled=False):
        self.enabled = enabled
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X, y=None):
        if not self.enabled:
            return self
        self.model.fit(X)
        return self

    def transform(self, X, Y=None):
        if not self.enabled:
            return X
        labels = self.model.predict(X)
        mask = (labels == 1)
        mask = np.asarray(mask).ravel()
        return X[mask]

    
class CreateFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lags: list, window_sizes: list):
        self.lags = lags
        self.window_sizes = window_sizes

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame=None):
        self.columns = X.columns
        return self
    

    def create_lagged_features(self, X:pd.DataFrame) -> pd.DataFrame:
        lagged_data = pd.DataFrame(index=X.index)
        lagged_features = {}
        for col in self.columns:
            for lag in self.lags:
                lagged_col_name = f"{col}_lag{lag}"
                lagged_features[lagged_col_name] = X[col].shift(lag)
        lagged_data = pd.concat([pd.DataFrame(lagged_features)], axis=1)
        return lagged_data

    def create_rolling_features(self, X:pd.DataFrame) -> pd.DataFrame:
        rolling_data = pd.DataFrame(index=X.index)
        for col in self.columns:
            for window in self.window_sizes:
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

    def transform(self, X: pd.DataFrame, Y: pd.DataFrame=None):
        lagged_features = self.create_lagged_features(X)
        rolling_features = self.create_rolling_features(X)
        cyc_features = self.create_cyc_features(X)
        combined_features = pd.concat([X, lagged_features, rolling_features, cyc_features], axis=1)
        return combined_features
# endregion

# region model training and evaluation functions
def __model_training_and_evaluation_functions__():
    pass
def searcher_builder(hyperspace: dict, model,len_train: int, iters=30, random_state=42, reg=True) -> BayesSearchCV:
    '''
    Build and return the hyperparameter searcher
    '''

    ps = PredefinedSplit(
        test_fold=[
            -1 if i < int((1 - VALIDATION_PORTION) * len_train) else 0
            for i in range(len_train)
        ]
    )

    pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('model', model)
    ])

    opt = BayesSearchCV(
        estimator=pipeline,  
        search_spaces=hyperspace,
        cv=ps,
        n_iter=iters,
        random_state=random_state,
        scoring='neg_root_mean_squared_error' if reg else 'accuracy',
    )
    return opt

def get_metrics(opt: BayesSearchCV, X_test, Y_test, Y_baseline, scaler_Y, column_name: str):
    '''
    Get metrics for the best model
    '''
    best_model = opt.best_estimator_
    Y_pred = best_model.predict(X_test)
    Y_pred = scaler_Y.inverse_transform(Y_pred)
    Y_test = scaler_Y.inverse_transform(Y_test)

    metrics = {}
    metrics['column'] = column_name
    metrics['Y_pred'] = Y_pred
    metrics['Y_test'] = Y_test
    metrics['best_params'] = opt.best_params_
    metrics['rmse'] = np.sqrt(np.mean((Y_pred - Y_test)**2))
    baseline_rmse=np.sqrt(np.mean((Y_baseline - Y_test)**2, axis=0))
    metrics['improvement_over_baseline_percentage'] = ((baseline_rmse - metrics['rmse'])/baseline_rmse*100)
    try:
        metrics['best_model_training_loss_curve'] = best_model['model'].train_loss_curve_
    except AttributeError:
        metrics['best_model_training_loss_curve'] = None
    return metrics

def get_classification_metrics(opt: BayesSearchCV, X_test, Y_test, Y_baseline, column_name: str):
    '''
    Get metrics for the best classification model
    '''
    from sklearn.metrics import accuracy_score, classification_report

    best_model = opt.best_estimator_
    Y_pred = best_model.predict(X_test)

    metrics = {}
    metrics['column'] = column_name
    metrics['Y_pred'] = Y_pred
    metrics['Y_test'] = Y_test
    metrics['best_params'] = opt.best_params_
    metrics['accuracy'] = accuracy_score(Y_test, Y_pred)
    baseline_accuracy = accuracy_score(Y_test, Y_baseline)
    metrics['improvement_over_baseline_percentage'] = ((metrics['accuracy'] - baseline_accuracy)/baseline_accuracy*100)
    metrics['classification_report'] = classification_report(Y_test, Y_pred)
    try:
        metrics['best_model_training_loss_curve'] = best_model['model'].train_loss_curve_
    except AttributeError:
        metrics['best_model_training_loss_curve'] = None
    return metrics
# endregion