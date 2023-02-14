import math
import time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from TaxiFareModel.trainer import Trainer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


def first_experiment(exp_name, n_rows):
    models_name = [LinearRegression(), RandomForestRegressor(), LinearSVR(), DecisionTreeRegressor(), Ridge()]
    for model in models_name:
        print(f"Modèle : {model.__class__.__name__}")
        trainer = Trainer(n_rows, model, experiment_name=exp_name)
        trainer.run()
        start = time.perf_counter()
        mse, rmse, r2 = trainer.evaluate()
        end = time.perf_counter()
        # logs the value of the n_rows parameter
        trainer.mlflow_log_param("n_rows", n_rows)
        trainer.mlflow_log_param("time", str(round(end-start, 3)) + "s")
        trainer.mlflow_log_param("model", model.__class__.__name__)
        trainer.mlflow_log_metric("RMSE", rmse)
        trainer.mlflow_log_metric("r2_score", r2)
        trainer.mlflow_log_metric("MSE", mse)
        trainer.save_model()


def advanced_model_experiment(exp_name, n_rows):
    models_name = [xgb.XGBRegressor(),
                   lgb.LGBMRegressor(), ExtraTreeRegressor()]
    for model in models_name:
        print(f"Modèle : {model.__class__.__name__}")
        trainer = Trainer(n_rows, model, experiment_name=exp_name)
        trainer.run()
        start = time.perf_counter()
        mse, rmse, r2 = trainer.evaluate()
        end = time.perf_counter()
        # logs the value of the n_rows parameter
        trainer.mlflow_log_param("n_rows", n_rows)
        trainer.mlflow_log_param("time", str(round(end-start, 3)) + "s")
        trainer.mlflow_log_param("model", model.__class__.__name__)
        trainer.mlflow_log_metric("RMSE", rmse)
        trainer.mlflow_log_metric("r2_score", r2)
        trainer.mlflow_log_metric("MSE", mse)



def xgboost_gridsearch(exp_name, n_rows):
    model = xgb.XGBRegressor()
    trainer = Trainer(n_rows, model, experiment_name=exp_name)
    # Define the hyperparameters to be tuned
    param_grid = {
        'model__n_estimators': [60, 70, 80 ],
        'model__learning_rate': [0.08, 0.1, 1.02],
        'model__max_depth': [3, 4, 5]
    }
    grid_search = GridSearchCV(trainer.pipe, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', refit=True)
    grid_search.fit(trainer.X_train, trainer.y_train)
    print(grid_search.best_params_)
    print("RMSE : ", math.sqrt(-grid_search.best_score_))
    trainer.mlflow_log_param("n_rows", n_rows)
    trainer.mlflow_log_param("model", model.__class__.__name__)
    trainer.mlflow_log_param("n_estimators", grid_search.best_params_[
                              'model__n_estimators'])
    trainer.mlflow_log_param("learning_rate", grid_search.best_params_[
                              'model__learning_rate'])
    trainer.mlflow_log_param("max_depth", grid_search.best_params_[
                              'model__max_depth'])
    trainer.mlflow_log_metric("rmse", math.sqrt(-grid_search.best_score_))
    trainer.save_model()


