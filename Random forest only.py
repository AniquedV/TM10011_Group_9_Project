#%%
import sys
print(sys.executable)
#%% Data loading functions. Uncomment the one you want to use
#from worcgist.load_data import load_data
#from worclipo.load_data import load_data
#from worcliver.load_data import load_data
from hn.load_data import load_data
#from ecg.load_data import load_data

data = load_data()
data = data.copy()  
print(f'The number of samples: {len(data.index)}')

print(f'The number of columns: {len(data.columns)}')

#%% Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest, skew, kurtosis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics
from sklearn.metrics import roc_auc_score

# Prepare data
data['label_bin'] = LabelEncoder().fit_transform(data['label'])

X = data.drop(columns=['label', 'label_bin']).values
y = data['label_bin'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#%% Define model
model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "min_samples_split": [2, 5, 10]
}

#%% GridSearch WITH cross-validation (only on training data!)
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv_inner,
    scoring='roc_auc',
    n_jobs=-1
)

# FIRST fit
grid.fit(X_train, y_train)

# THEN get best model
best_model = grid.best_estimator_

print("Best parameters:", grid.best_params_)

#%% Final evaluation on TEST set (20%)
y_test_probs = best_model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_test_probs)

print("Final Test AUC:", auc_test)

