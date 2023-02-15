from functions import helper
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import xgboost as xgb
import lightgbm as lgb


def main():
    standard_models = [LinearRegression(), RandomForestRegressor(
    ), LinearSVR(), DecisionTreeRegressor(), Ridge()]
    advanced_models = [xgb.XGBRegressor(), lgb.LGBMRegressor(),
                       ExtraTreeRegressor()]
    helper.launch_experiments(standard_models, "Standard Models", 1000)
    helper.launch_experiments(advanced_models, "Advanced Models", 1000)
    helper.xgboost_hyperparam("Grid Search", 5000)
    # helper.xgboost_hyperparam("Random Search", 5000)
    # helper.xgboost_hyperparam("Bayesian Optimisation", 5000)


if __name__ == "__main__":
    main()
