from functions import helper


def main():
    helper.first_experiment("Standard Models", 5000)
    helper.advanced_model_experiment("Advanced Models", 5000)
    helper.xgboost_gridsearch("Hyper paramétrage XGBoost", 5000)
    

if __name__ == "__main__":
    main()
