
import pandas as pd
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier


from preprocessing import get_features_pipeline


def run_gridsearch(
        df,
        numeric_features, 
        categorical_features,
        clf,
        param_grid,
        outfilename):
    
    # Build preprocessing pipeline
    features_pipeline = get_features_pipeline(numeric_features, categorical_features)
    X = features_pipeline.fit_transform(df)
    y = df['Survived']


    # Set up the randomized search
    gs = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    # Fit
    print("Running randomized search...")
    gs.fit(X, y)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X)
    
    print("Best Parameters:")
    print(gs.best_params_)
    print(f"Best Accuracy: {gs.best_score_:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred))

    # Save the model
    joblib.dump(best_model, outfilename)
    print(f"Model saved as {outfilename}")

    return features_pipeline, gs

    

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    y = df['Survived']

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

    features_pipeline, gs = run_gridsearch(
        df, numeric_features, categorical_features,
        xgb_clf,
        param_distributions,  
        'model/best_xgb_serach_model.pkl'
        )


    

