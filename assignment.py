

#%% Data loading functions. Uncomment the one you want to use
#from worcgist.load_data import load_data
#from worclipo.load_data import load_data
#from worcliver.load_data import load_data
from hn.load_data import load_data
#from ecg.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')

print(f'The number of columns: {len(data.columns)}')

#%% Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn import model_selection, metrics

import warnings
warnings.filterwarnings('ignore')

#%% Helper functions

def preprocess_data(X):
    X = X.copy()
    
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
    X = X.mask(outliers, np.nan)

    X = X.fillna(X.median())
    return X


def plot_learning_curve(estimator, title, X, y, axes, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt


def plot_roc_curve(y_score, y_truth):
    y_score = y_score[:, 1]
    fpr, tpr, _ = roc_curve(y_truth, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


#%% Data preparation

data['label_bin'] = LabelEncoder().fit_transform(data['label'])
print(data[['label', 'label_bin']].head())

X = data.drop(columns=['label', 'label_bin'])
y = data['label_bin']

# Preprocess and split
X_cleaned = preprocess_data(X)

X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(
    X_cleaned, y.values, test_size=0.2, random_state=42, stratify=y.values
)

# Ensure they are numpy arrays (not pandas Series)
y_train_cleaned = np.asarray(y_train_cleaned)
y_test_cleaned = np.asarray(y_test_cleaned)

# Scale train/test
scaler_cleaned = RobustScaler()
X_train = scaler_cleaned.fit_transform(X_train_cleaned)
X_test = scaler_cleaned.transform(X_test_cleaned)

# Setup scalers for different model evaluations
scaler_full = RobustScaler()
X_scaled_full = scaler_full.fit_transform(X)
y_array = y.values

cv_10fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

#%% SVM
print("\n--- SVM ---")

# SVM hyperparameter tuning
models_and_grids = [
    ("linear", SVC(probability=True), {
        "kernel": ["linear"],
        "C": [0.1, 1, 10]
    }),
    ("rbf", SVC(probability=True), {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1]
    }),
    ("poly", SVC(probability=True), {
        "kernel": ["poly"],
        "C": [0.1, 1, 10],
        "degree": [2, 3, 5],
        "coef0": [0, 1]
    })
]

best_svm_models = []

for name, model, param_grid in models_and_grids:
    print(f"\nTuning {name} SVM")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_10fold,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_scaled_full, y_array)

    print("Best parameters:", grid.best_params_)
    best_svm_models.append((name, grid.best_params_))

    results = []
    for train_index, test_index in cv_10fold.split(X_scaled_full, y_array):
        X_cv_train, X_cv_test = X_scaled_full[train_index], X_scaled_full[test_index]
        y_cv_train, y_cv_test = y_array[train_index], y_array[test_index]

        best_model = grid.best_estimator_
        best_model.fit(X_cv_train, y_cv_train)

        y_probs = best_model.predict_proba(X_cv_test)[:, 1]
        auc_score = metrics.roc_auc_score(y_cv_test, y_probs)
        results.append({'auc': auc_score, 'set': 'test'})

        y_train_probs = best_model.predict_proba(X_cv_train)[:, 1]
        auc_train = metrics.roc_auc_score(y_cv_train, y_train_probs)
        results.append({'auc': auc_train, 'set': 'train'})

    results_df = pd.DataFrame(results)
    sns.boxplot(y='auc', x='set', data=results_df)
    plt.title(f"Tuned {name} SVM")
    plt.show()

# SVM uses raw X with fresh scaler
clsfs = [
    ("SVM linear", SVC(kernel='linear', C=0.1, gamma='scale', probability=True)),
    ("SVM poly",   SVC(kernel='poly', C=10, coef0=1, degree=2, gamma='scale', probability=True)),
    ("SVM rbf",    SVC(kernel='rbf', C=10, gamma='scale', probability=True))
]

svmlin = SVC(kernel='linear', gamma='scale', probability=True)
svmrbf = SVC(kernel='rbf', gamma='scale', probability=True)
svmpoly = SVC(kernel='poly', degree=3, gamma='scale', probability=True)

print("\nLearning curves:")
fig, ax = plt.subplots(figsize=(8, 6))
plot_learning_curve(svmlin, "Learning Curve: SVM Linear", X_train, y_train_cleaned, ax, cv=10)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
plot_learning_curve(svmpoly, "Learning Curve: SVM Poly", X_train, y_train_cleaned, ax, cv=10)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
plot_learning_curve(svmrbf, "Learning Curve: SVM RBF", X_train, y_train_cleaned, ax, cv=10)
plt.show()

for name, clf in clsfs:
    clf.fit(X_train, y_train_cleaned)

    y_test_pred = clf.predict(X_test)
    y_test_probs = clf.predict_proba(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test_cleaned, y_test_pred).ravel()

    print(f"\n{name}")
    print("accuracy:", accuracy_score(y_test_cleaned, y_test_pred))
    print("sensitivity:", recall_score(y_test_cleaned, y_test_pred))
    print("specificity:", tn / (tn + fp))

    plot_roc_curve(y_test_probs, y_test_cleaned)

#%% KNN
print("\n--- KNN ---")

param_grid = {"n_neighbors": list(range(1, 26, 2))}
k_list = list(range(1, 26, 2))

best_n_neighbors = []
all_train = []
all_test = []

for train_index, test_index in cv_10fold.split(X_train, y_train_cleaned):

    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train_cleaned[train_index], y_train_cleaned[test_index]

    knn = KNeighborsClassifier()
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        knn,
        param_grid,
        cv=cv_inner,
        scoring='roc_auc',
        return_train_score=True,
        n_jobs=-1
    )

    grid_search.fit(X_cv_train, y_cv_train)

    clf = grid_search.best_estimator_
    best_n_neighbors.append(clf.n_neighbors)

    mean_train_scores = grid_search.cv_results_["mean_train_score"]
    mean_test_scores = grid_search.cv_results_["mean_test_score"]

    all_train.append(mean_train_scores)
    all_test.append(mean_test_scores)

all_train = np.array(all_train)
all_test = np.array(all_test)

train_mean = all_train.mean(axis=0)
train_std = all_train.std(axis=0)

test_mean = all_test.mean(axis=0)
test_std = all_test.std(axis=0)

optimal_n = k_list[np.argmax(test_mean)]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.grid()

ax.fill_between(k_list,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.1, color="r")

ax.fill_between(k_list,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.1, color="g")

ax.plot(k_list, train_mean, 'o-', color="r", label="Training score")
ax.plot(k_list, test_mean, 'o-', color="g", label="Cross Validation score")

ax.set_xlabel("Number of neighbors (k)")
ax.set_ylabel("AUC score")
ax.set_title("KNN Learning Curve (GridSearchCV)")
ax.legend()

plt.show()

print(f"The optimal N = {optimal_n}")

best_knn = KNeighborsClassifier(n_neighbors=optimal_n)
best_knn.fit(X_train, y_train_cleaned)

score_train = best_knn.score(X_train, y_train_cleaned)
score_test = best_knn.score(X_test, y_test_cleaned)

y_pred_proba = best_knn.predict_proba(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test_cleaned, y_pred_proba[:, 1])
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw=2
plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


print(f"Training accuracy: {score_train:.3f}")
print(f"Test accuracy: {score_test:.3f}")

#%% Random Forest
print("\n--- Random Forest ---")

model = RandomForestClassifier(random_state=42)

# Define hyperparameter distributions for RandomizedSearchCV
param_distributions = {
    "n_estimators": randint(50, 150),
    "max_depth": randint(2, 5),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(5, 10),
}

cv_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring='roc_auc',
    cv=cv_rf,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Tuning Random Forest with RandomizedSearchCV...")
random_search.fit(X_train, y_train_cleaned)

best_rf = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

fig, ax = plt.subplots(figsize=(7,5))

plot_learning_curve(
    estimator=best_rf,
    title="Learning Curve (Random Forest)",
    X=X_train,
    y=y_train_cleaned,
    axes=ax,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.grid()
plt.show()

y_test_probs = best_rf.predict_proba(X_test)
auc_test = roc_auc_score(y_test_cleaned, y_test_probs[:, 1])

print("Final Test AUC:", auc_test)

y_test_pred = best_rf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test_cleaned, y_test_pred).ravel()
accuracy = accuracy_score(y_test_cleaned, y_test_pred)
sensitivity = recall_score(y_test_cleaned, y_test_pred)
specificity = tn / (tn + fp)
print("accuracy:", accuracy)
print("sensitivity:", sensitivity)
print("specificity:", specificity)

plot_roc_curve(y_test_probs, y_test_cleaned)

#%% XGBoost
print("\n--- XGBoost ---")

# XGBoost uses a Pipeline with scaling inside
pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("model", XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        gamma=1.0,
        random_state=42,
        eval_metric="logloss"
    ))
])

param_grid = {
    "model__max_depth": [3, 4],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.6, 0.8],
    "model__colsample_bytree": [0.6, 0.8]
}

cv_xgb = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv_xgb,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0
)

print("XGBoost - tuning with GridSearchCV...")
grid_search.fit(X_train_cleaned, y_train_cleaned)

print("Best parameters:")
print(grid_search.best_params_)

best_xgb = grid_search.best_estimator_

def plot_learning_curve_xgb(estimator, title, X, y, 
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
        cv=cv,
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

plot_learning_curve_xgb( 
    best_xgb, 
    title="Learning Curve - XGBoost", 
    X=X_train_cleaned, 
    y=y_train_cleaned, 
    axes=ax, 
    cv=cv_xgb, 
    n_jobs=-1, 
    scoring='accuracy' ) 
plt.show()

cv_scores = cross_val_score(
    best_xgb,
    X_train_cleaned,
    y_train_cleaned,
    cv=cv_xgb,
    scoring="roc_auc"
)

print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

best_xgb.fit(X_train_cleaned, y_train_cleaned)

y_pred = best_xgb.predict(X_test_cleaned)
acc = accuracy_score(y_test_cleaned, y_pred)

print(f"Test Accuracy: {acc:.3f}")

y_proba = best_xgb.predict_proba(X_test_cleaned)[:, 1]

fpr, tpr, _ = roc_curve(y_test_cleaned, y_proba)
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





