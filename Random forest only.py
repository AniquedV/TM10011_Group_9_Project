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
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest, skew, kurtosis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics

# Prepare data
data['label_bin'] = LabelEncoder().fit_transform(data['label'])

X = data.drop(columns=['label', 'label_bin']).values
y = data['label_bin'].values

#%%
# Define model + params
model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200], #number of trees; vaak voorkomende hoeveelheden --> hoger ontplofd mn laptop
    "max_depth": [3, 5, 10],        #max amount of decisions in each tree --> vooral te maken met fitting. Te veel = overfitting, te weinig = underfitting
    "min_samples_leaf": [1, 3, 5],  #samples (choices) in a leaf
    "min_samples_split": [2, 5, 10] #samples in a split --> min aantal samples dat nodig is om een split te maken, haalt zwakke trees er uit.
}

# Nested Cross-Validation (now on whole dataset, but in full model only on training)
cv_outer = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []
best_params_list = []

for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner CV (hyperparameter tuning)
    cv_inner = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = model_selection.GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_inner,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params_list.append(grid.best_params_)

    print(f"Fold {fold} best params: {grid.best_params_}")

    # Test performance
    y_test_probs = best_model.predict_proba(X_test)[:, 1]
    auc_test = metrics.roc_auc_score(y_test, y_test_probs)

    # Train performance
    y_train_probs = best_model.predict_proba(X_train)[:, 1]
    auc_train = metrics.roc_auc_score(y_train, y_train_probs)

    results.append({"fold": fold, "set": "test", "auc": auc_test})
    results.append({"fold": fold, "set": "train", "auc": auc_train})

# Results analysis
results_df = pd.DataFrame(results)
best_params_df = pd.DataFrame(best_params_list)

print("\nBest hyperparameters per fold:")
print(best_params_df)

print("\nMost frequent best params:") #We select the most frequent combination of hyperparameters
print(best_params_df.value_counts())

print("\nMean AUC scores:")
print(results_df.groupby("set")["auc"].mean())

# Visualization
plt.figure(figsize=(6, 5))
sns.boxplot(x="set", y="auc", data=results_df)
plt.title("Random Forest Performance (Nested CV)")
plt.show()

#%%
# Selecting most frequent hyperparameters

final_params = best_params_df.value_counts().idxmax()  # most frequent combination
final_params = dict(zip(best_params_df.columns, final_params)) 

print("Final selected hyperparameters (most frequent):")
print(final_params)

# Train final model on full dataset (nog aanpassen naar 80% training set)
final_model = RandomForestClassifier(
    n_estimators=final_params['n_estimators'],
    max_depth=final_params['max_depth'],
    random_state=42
)

final_model.fit(X, y)  # train on all data
print("\nFinal model trained on full dataset.")

# Final performance with CV, --> wordt dus getest op laatste 20%

from sklearn.model_selection import cross_val_score

auc_scores = cross_val_score(
    final_model,
    X,
    y,
    cv=10,
    scoring='roc_auc',
    n_jobs=-1
)

print("\nEstimated AUC on unseen data (10-fold CV):")
print(f"Mean AUC: {auc_scores.mean():.3f}")
print(f"Std AUC: {auc_scores.std():.3f}")

# Plot
mean_auc = auc_scores.mean()
std_auc = auc_scores.std()

auc_df = pd.DataFrame({
    "set": ["10-fold CV"],
    "mean_auc": [mean_auc],
    "std_auc": [std_auc]
})

plt.figure(figsize=(5,5))
sns.barplot(x="set", y="mean_auc", data=auc_df, yerr=auc_df["std_auc"], palette="viridis")
plt.ylim(0, 1)
plt.ylabel("AUC")
plt.title("Final Random Forest AUC (10-fold CV)")
plt.show()
# %%
