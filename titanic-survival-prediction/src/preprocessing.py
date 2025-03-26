
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Cabin_letter'] = df['Cabin'].str[0].fillna('U')
        df.drop('Cabin', axis=1, inplace=True)

        df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer',
            'Major': 'Officer', 'Capt': 'Officer',
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Don': 'Royalty', 'Dona': 'Royalty', 'Sir': 'Royalty',
            'Lady': 'Royalty', 'Countess': 'Royalty', 'Jonkheer': 'Royalty'
        }
        df['Title'] = df['Title'].map(title_mapping).fillna('Unknown')
        df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z]+)', expand=False).fillna('Numeric')
        df['TicketCount'] = df.groupby('Ticket')['Ticket'].transform('count')
        df['IsSharedTicket'] = (df['TicketCount'] > 1).astype(int)

        return df


def get_features_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    columns_transformer = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    features_pipeline = Pipeline([
        ('df_preprocess', DataFramePreprocessor()),
        ('columns', columns_transformer)
    ])
    return features_pipeline


if __name__ == "__main__":
    pass