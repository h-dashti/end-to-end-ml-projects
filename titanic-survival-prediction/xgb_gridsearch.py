
import pandas as pd
import os
from xgboost import XGBClassifier
import joblib

from src import utils

empty_praram_grid = False


if __name__ == '__main__':

    df = pd.read_csv('data/train.csv')

    # Define features
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    categorical_features = [
        'Pclass', 'Sex', 'Embarked', 'Cabin_letter', 'Title',
        'IsAlone', 'TicketPrefix', 'IsSharedTicket'
    ]

    param_grid = {
        'n_estimators': [50, 100, 300],                # Default: 100
        'max_depth': [3, 5, 6, 7, 9],                            # Default: 6
        'learning_rate': [0.01, 0.1, 0.2, 0.3],            # Default: 0.3
        'subsample': [0.6, 0.8, 1.0],                  # Default: 1.0
        'colsample_bytree': [0.6, 0.8, 1.0],           # Default: 1.0
        'gamma': [0, 0.1, 0.5, 1, 5],                            # Default: 0
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1],                     # Default: 0
        'reg_lambda': [0.5, 1, 1.5, 5],                       # Default: 1
        'min_child_weight': [1, 3, 5, 7, 10],                    # Default: 1
        'scale_pos_weight': [1, 2, 5]                            # Default: 1 (important if dataset is imbalanced)
    }

    if empty_praram_grid:
        param_grid = {}

    

    # DEFINE the model
    clf = XGBClassifier(eval_metric='logloss', random_state=42,)
    features_pipeline, gs = utils.run_gridsearch(
        df, numeric_features, categorical_features, clf,param_grid,  )


    # PRINT the results
    print("Best Parameters:")
    print(gs.best_params_)
    print(f"Best Accuracy: {gs.best_score_:.4f}")

    # SAVE the model
    odir = 'models/'+ utils.get_current_datatime()
    fname = "best_xgb_serach_model"
    os.makedirs(odir, exist_ok=True)
    outfilename = f'{odir}/{fname}.joblib'
    joblib.dump(gs.best_estimator_, outfilename)
    print(f"Model saved as {outfilename}")

    fname = "pipeline_" + fname
    outfilename = f'{odir}/{fname}.joblib'
    joblib.dump(features_pipeline, outfilename)
    print(f"Pipeline saved as {outfilename}")
    

