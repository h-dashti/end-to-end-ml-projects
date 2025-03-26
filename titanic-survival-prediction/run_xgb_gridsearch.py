
import pandas as pd
import os
from xgboost import XGBClassifier

from src import utils



if __name__ == '__main__':

    df = pd.read_csv('data/train.csv')

    # Define features
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    categorical_features = [
        'Pclass', 'Sex', 'Embarked', 'Cabin_letter', 'Title',
        'IsAlone', 'TicketPrefix', 'IsSharedTicket'
    ]

     # Define the parameter distributions for RandomizedSearchCV
    param_distributions = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [2, 3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
        'gamma': [0, 1, 5, 10]
    }

    

    # Define the model
    xgb_clf = XGBClassifier(eval_metric='logloss', random_state=42,
                            # use_label_encoder=False, 
                            )

    os.makedirs('model', exist_ok=True)

    features_pipeline, gs = utils.run_gridsearch(
        df, numeric_features, categorical_features,
        xgb_clf,
        param_distributions,  
        'model/best_xgb_serach_model.pkl'
        )


    

