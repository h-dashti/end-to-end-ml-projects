from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

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
    
