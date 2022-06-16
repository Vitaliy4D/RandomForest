
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('C:/ML/Housing_Prices_Competition/train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
# need to test imputing with kNN
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=150, random_state=0,max_depth=18,
                                 criterion='squared_error', min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                 max_features=27,max_leaf_nodes=1000,min_impurity_decrease=0.01,
                                 bootstrap=False)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))


### to choose best model Parameters and to learn how each one influence the result
# create function and iterate through each of Parameter

def get_score(n):   
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='mean')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define model
    model = RandomForestRegressor(n_estimators=n, random_state=0)

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)
                         ])

    # Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

    preds = clf.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
    # print('MAE:', mean_absolute_error(y_valid, preds))


# example to check what is the best n_estimators
n=list(range(50,700,50))
print(n)

# add to a dict name of n and the result of RandomForest with this n
f={}
for i in n:
    name=i
    f.update({name:get_score(i)})
print(f)

# let's plot results to see how parameter influences the model result
plt.plot(f.keys(),f.values())


# later on will add searching best parameters with Gridsearch

from sklearn.model_selection import GridSearchCV

param_grid = {'max_leaf_nodes': [ 60, 150, 400],
'min_samples_leaf': [2, 3, 4],
'max_features': [4, 13, 20]
}

gs_cv.best_params_










