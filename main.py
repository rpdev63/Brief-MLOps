from functions import helper


def main():
    # helper.first_experiment("Standard Models", 1000)
    # helper.advanced_model_experiment("Advanced Models", 10000)
    helper.xgboost_hyperparameters_tuning("Hyper paramétrage XGBoost", 1000)


if __name__ == "__main__":
    main()
