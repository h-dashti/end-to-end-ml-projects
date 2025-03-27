from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from datetime import datetime

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
        ):
    
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
        verbose=2,
        n_jobs=-1,
    )

    # Fit
    print("Running grid search...")
    gs.fit(X, y)

    return features_pipeline, gs

def get_current_datatime() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    return timestamp