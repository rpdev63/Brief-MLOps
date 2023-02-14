import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data
from TaxiFareModel.utils import compute_rmse
from functools import cached_property
from mlflow.tracking import MlflowClient


class Trainer:
    def __init__(self, n_rows, model, experiment_name) -> None:
        self.df = get_data(nrows=n_rows)
        self.y = self.df['fare_amount']
        self.X = self.df.drop(columns=['fare_amount'], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42)
        self.set_pipeline(model)
        self.experiment_name = experiment_name
        self.mlflow_client = MlflowClient()

    def set_pipeline(self, model):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ("ohe", OneHotEncoder(handle_unknown='ignore'))
        ])

        self.pipe = Pipeline([
            ('features', ColumnTransformer(transformers=[
                ('distance', dist_pipe, [
                 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']),
                ('time', time_pipe, ['pickup_datetime'])
            ])),
            ('model', model)
        ])

    def run(self):
        self.pipe.fit(self.X_train, self.y_train)

    def evaluate(self):
        '''returns the value of the RMSE'''
        # A COMPLETER
        y_pred = self.pipe.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        print(rmse)
        return rmse

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    @cached_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @cached_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        path = 'models/'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + \
            f"{self.pipe.named_steps['model'].__class__.__name__}.joblib"
        joblib.dump(self.pipe, filename)
