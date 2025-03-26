from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import joblib

from . import preprocessing

def learning_cuvre_plot(model, X, y, ax=None, title=None, outfilename=None):

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        )
    if ax:
        train_scores_mean = train_scores.mean(axis=1)
        val_scores_mean = val_scores.mean(axis=1)

        ax.plot(train_sizes, train_scores_mean, label='Training Accuracy', marker='o')
        ax.plot(train_sizes, val_scores_mean, label='Validation Accuracy', marker='s')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Learning Curve - {title}')
        ax.legend()
        ax.grid(True)
        if outfilename:
            plt.savefig(outfilename, bbox_inches='tight', )
    

def run_gridsearch(
        df,
        numeric_features, 
        categorical_features,
        clf,
        param_grid,
        outfilename):
    
    # Build preprocessing pipeline
    features_pipeline = preprocessing.get_features_pipeline(numeric_features, categorical_features)
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
