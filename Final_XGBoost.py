
# Import the dataset and load it into a pandas DataFrame
from hn.load_data import load_data
data = load_data()


# Print the number of rows and colums
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

# =====================
# Imports
# =====================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import shapiro, normaltest, skew, kurtosis

from xgboost import XGBClassifier


# =====================
# Data preparation
# =====================
data['label_bin'] = LabelEncoder().fit_transform(data['label']) #convert T12 and T34 into binary numbers

X = data.drop(columns=['label', 'label_bin']) #drop label to only be left with the features
y = data['label_bin'] # make y with only the binary labels


def preprocess_data(X):
    X = X.copy()

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
    X = X.mask(outliers, np.nan)

    X = X.fillna(X.median())

    return X

X_cleaned = preprocess_data(X)
#%%

scaler = RobustScaler()
X2 = scaler.fit_transform(X_cleaned)
y2 = y.values



# Split dataset into a training (80%) and test (20%) set
X_train, X_test, y_train, y_test = train_test_split(
    X2, y2,
    test_size=0.2,
    random_state=42, # ensures reproducibility by splitting it at the same place each time
    stratify=y2 # This keeps the class distribution identical in train and test set
)

# Cross-validation, strafifiedkfold makes sure each fold has the same class proportions
cv_inner = StratifiedKFold(n_splits=10, 
                           shuffle=True, #improves randomness of folds
                           random_state=42)

# =====================
# Pipeline
# =====================

# The pipeline ensures that scaling happens inside the CV folds
pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("model", XGBClassifier(
        n_estimators=100,       #number of trees
        max_depth=2,            #depth of trees (small to reduce overfitting)
        learning_rate=0.01,     #small step size to result in better generalisation and slower learning
        subsample=0.7,          #70% of samples per tree to reduce overfitting
        colsample_bytree=0.7,   #70% of features per tree to add randomness and reduce overfitting
        reg_alpha=1.0,          #L1 regularization
        reg_lambda=5.0,         #L2 regularization
        gamma=1.0,              #how much improvement is needed to make a split in a decision tree
        random_state=42,        #make it reproducable
        eval_metric="logloss"   #loss function is used because we have binary classification
    ))
])

# =====================
# GridSearch (hyperparameter tuning)
# =====================
param_grid = {
    "model__max_depth": [3, 4],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.6, 0.8],
    "model__colsample_bytree": [0.6, 0.8]
}


grid_search = GridSearchCV(     # trying all combinations of parameters
    estimator=pipe,
    param_grid=param_grid,
    cv=cv_inner,
    scoring="roc_auc",
    n_jobs=-1,                  #use all the available cores
    verbose=1                   #show how many fits needs to be done
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# =====================
# Learning curve
# =====================
def plot_learning_curve(estimator, title, X, y, 
                        axes, 
                        ylim=None, 
                        cv=None, 
                        n_jobs=None, 
                        train_sizes=np.linspace(0.1,1.0,5),
                        scoring='accuracy'):
    axes.set_title(title)

    if ylim is not None:
        axes.set_ylim(*ylim)

    axes.set_xlabel("Training examples")
    axes.set_ylabel("score")


    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv_inner,
        scoring=scoring,
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores, axis=1)

    axes.grid()

    axes.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color = "r"
    )

    axes.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha = 0.1,
        color = "g"
    )

    axes.plot(train_sizes, train_mean, 'o-', color="r",
              label = "Training score")
    axes.plot(train_sizes, test_mean, 'o-', color="g",
              label = "Cross-validation score")
    
    axes.legend(loc="best")

    return axes
fig, ax = plt.subplots(figsize=(7,5))

plot_learning_curve( 
    best_model, 
    title="Learning Curve - XGBoost", 
    X=X_train, 
    y=y_train, 
    axes=ax, 
    cv=cv_inner, 
    n_jobs=-1, 
    scoring='accuracy' ) 
plt.show()
    
# =====================
# Cross-validation score
# =====================

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=cv_inner,
    scoring="roc_auc"
)

print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# =====================
# Test performance
# =====================
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {acc:.3f}")

# =====================
# ROC curve
# =====================
y_proba = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.grid()
plt.show()

print(f"Test AUC: {roc_auc:.3f}")
