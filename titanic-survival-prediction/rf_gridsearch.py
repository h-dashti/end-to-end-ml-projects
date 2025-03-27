
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

from src import utils

empty_praram_grid = True


if __name__ == '__main__':

    df = pd.read_csv('data/train.csv')

    # Define features
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    categorical_features = [
        'Pclass', 'Sex', 'Embarked', 'Cabin_letter', 'Title',
        'IsAlone', 'TicketPrefix', 'IsSharedTicket'
    ]

    param_grid = {
        'n_estimators': [100, 200, 300, 500],              # Default: 100
        'max_depth': [None, 5, 10, 15, 20],                # Default: None (nodes expanded until all leaves are pure)
        'min_samples_split': [2, 5, 10],                   # Default: 2
        'min_samples_leaf': [1, 2, 4],                     # Default: 1
        'max_features': ['auto', 'sqrt', 'log2'],          # Default: 'auto' (deprecated in newer versions)
        'bootstrap': [True, False],                        # Default: True
        'criterion': ['gini', 'entropy', 'log_loss'],      # Default: 'gini' (log_loss from sklearn 1.1+)
    }

    if empty_praram_grid:
        param_grid = {}



    # Define the model
    clf = RandomForestClassifier(random_state=42)
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
    


    

