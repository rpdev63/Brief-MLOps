import math
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from TaxiFareModel.trainer import Trainer
import xgboost as xgb
from skopt import BayesSearchCV


def launch_experiments(models, exp_name, n_rows):
    for model in models:
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


def xgboost_hyperparam(exp_name, n_rows):
    model = xgb.XGBRegressor()
    trainer = Trainer(n_rows, model, experiment_name=exp_name)
    # Define the hyperparameters to be tuned
    param_grid = {
        'model__n_estimators': [60, 70, 80],
        'model__learning_rate': [0.08, 0.1, 1.02],
        'model__max_depth': [3, 4, 5]
    }
    param_dist = {
        'model__n_estimators': range(50, 100),
        'model__learning_rate': [0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0],
        'model__max_depth': range(3, 6)
    }
    pbounds = {
        'model__learning_rate': (0.01, 0.5),
        'model__max_depth': (3, 9),
        'model__min_child_weight': (1, 5),
        'model__gamma': (0, 1),
        'model__n_estimators': (50, 150),
    }
    match exp_name:
        case "Grid Search":
            grid_search = GridSearchCV(trainer.pipe, param_grid=param_grid,
                                       cv=5, scoring='neg_mean_squared_error', refit=True)
            grid_search.fit(trainer.X_train, trainer.y_train)
            best_params, rmse = grid_search.best_params_, math.sqrt(-grid_search.best_score_)
        case "Random Search":
            random_search = RandomizedSearchCV(
                trainer.pipe, param_distributions=param_dist, n_iter=50, scoring='neg_mean_squared_error', cv=5, refit=True)
            random_search.fit(trainer.X_train, trainer.y_train)
            best_params, rmse = random_search.best_params_, math.sqrt(-random_search.best_score_)
        case "Bayesian Optimisation":
            bayes_opt = BayesSearchCV(
                trainer.pipe, pbounds, n_iter=50, scoring='neg_mean_squared_error', cv=5, refit=True)
            bayes_opt.fit(trainer.X_train, trainer.y_train)
            best_params, rmse = bayes_opt.best_params_, math.sqrt(-bayes_opt.best_score_)
        case _: 
            print("Error")

    print("Best parameters : ", best_params)
    print("RMSE : ", rmse)
    trainer.mlflow_log_param("n_rows", n_rows)
    trainer.mlflow_log_param("model", model.__class__.__name__)
    trainer.mlflow_log_param("n_estimators", best_params[
        'model__n_estimators'])
    trainer.mlflow_log_param("learning_rate", best_params[
        'model__learning_rate'])
    trainer.mlflow_log_param("max_depth", best_params[
        'model__max_depth'])
    trainer.mlflow_log_metric("rmse", rmse)
    trainer.save_model()

