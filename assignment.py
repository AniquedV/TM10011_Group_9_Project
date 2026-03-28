

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
import seaborn

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
from sklearn import feature_selection

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
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores  = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
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
    '''
    Plot an ROC curve.
    '''
    # Only take scores for class = 1
    y_score = y_score[:, 1]

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_truth, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
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

# Heatmap of possible missing data
X_zeros = X.replace(0.0, np.nan)  

col_missing_counts = (X == 0.0).sum()
col_missing_counts_sorted = col_missing_counts.sort_values(ascending=False)
print(col_missing_counts_sorted)

# HEATMAP PLOTTEN
plt.figure(figsize=(12,6))
seaborn.heatmap(X_zeros.isnull(), cbar=False, xticklabels=False, yticklabels=False)
plt.title("Missing data heatmap")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()

# Preprocess and split
X_cleaned = preprocess_data(X)

X_train_cleaned, X_test_cleaned, y_train, y_test = train_test_split(
    X_cleaned, y.values, test_size=0.2, random_state=42, stratify=y.values
)

# Scale train/test
scaler_cleaned = RobustScaler()
X_train = scaler_cleaned.fit_transform(X_train_cleaned)
X_test = scaler_cleaned.transform(X_test_cleaned)


cv_10fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

#%% SVM
print("\n--- SVM ---")

# Store results for summary
results_summary = []

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

    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    best_svm_models.append((name, grid.best_params_))

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
plot_learning_curve(svmlin, "Learning Curve: SVM Linear", X_train, y_train, ax, cv=10)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
plot_learning_curve(svmpoly, "Learning Curve: SVM Poly", X_train, y_train, ax, cv=10)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
plot_learning_curve(svmrbf, "Learning Curve: SVM RBF", X_train, y_train, ax, cv=10)
plt.show()

for name, clf in clsfs:
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)
    y_test_probs = clf.predict_proba(X_test)
    
    auc_test = roc_auc_score(y_test, y_test_probs[:, 1])

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    accuracy = accuracy_score(y_test, y_test_pred)
    sensitivity = recall_score(y_test, y_test_pred)
    specificity = tn / (tn + fp)

    print(f"\n{name}")
    print(f"AUC: {auc_test:.3f}")
    print(f"accuracy: {accuracy:.3f}")
    print(f"sensitivity: {sensitivity:.3f}")
    print(f"specificity: {specificity:.3f}")
    
    # Store results
    results_summary.append({
        'Model': name,
        'AUC': auc_test,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    })

    plot_roc_curve(y_test_probs, y_test)

#%% KNN
print("\n--- KNN ---")

# Feature selection with RFECV on training data
rfecv = feature_selection.RFECV(
    estimator=SVC(kernel='linear'), 
    step=1,
    cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc')
rfecv.fit(X_train, y_train)

scores = rfecv.cv_results_["mean_test_score"]

scores_no_first = scores[1:]
best_index = scores_no_first.argmax() +1
best_score = scores[best_index]
best_n_features = best_index + 1

print(best_n_features, best_score)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
plt.show()


param_grid = {"n_neighbors": list(range(1, 26, 2))}
k_list = list(range(1, 26, 2))

best_n_neighbors = []
all_train = []
all_test = []

for train_index, test_index in cv_10fold.split(X_train, y_train):

    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]

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
best_knn.fit(X_train, y_train)

y_pred_proba = best_knn.predict_proba(X_test)
y_pred = best_knn.predict(X_test)

auc_test = roc_auc_score(y_test, y_pred_proba[:, 1])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)

plot_roc_curve(y_pred_proba, y_test)


print(f"\nKNN")
print(f"AUC: {auc_test:.3f}")
print(f"accuracy: {accuracy:.3f}")
print(f"sensitivity: {sensitivity:.3f}")
print(f"specificity: {specificity:.3f}")

# Store results
results_summary.append({
    'Model': 'KNN',
    'AUC': auc_test,
    'Accuracy': accuracy,
    'Sensitivity': sensitivity,
    'Specificity': specificity
})

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
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

fig, ax = plt.subplots(figsize=(7,5))

plot_learning_curve(
    estimator=best_rf,
    title="Learning Curve (Random Forest)",
    X=X_train,
    y=y_train,
    axes=ax,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.grid()
plt.show()

y_test_probs = best_rf.predict_proba(X_test)
auc_test = roc_auc_score(y_test, y_test_probs[:, 1])

y_test_pred = best_rf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
accuracy = accuracy_score(y_test, y_test_pred)
sensitivity = recall_score(y_test, y_test_pred)
specificity = tn / (tn + fp)

print(f"\nRandom Forest Test Metrics:")
print(f"AUC: {auc_test:.3f}")
print(f"accuracy: {accuracy:.3f}")
print(f"sensitivity: {sensitivity:.3f}")
print(f"specificity: {specificity:.3f}")

# Store results
results_summary.append({
    'Model': 'Random Forest',
    'AUC': auc_test,
    'Accuracy': accuracy,
    'Sensitivity': sensitivity,
    'Specificity': specificity
})

plot_roc_curve(y_test_probs, y_test)

#%% XGBoost
print("\n--- XGBoost ---")

# XGBoost (data already scaled)
pipe = Pipeline([
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
grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

best_xgb = grid_search.best_estimator_

fig, ax = plt.subplots(figsize=(7,5))

plot_learning_curve(
    best_xgb, 
    title="Learning Curve - XGBoost", 
    X=X_train, 
    y=y_train, 
    axes=ax, 
    cv=cv_xgb, 
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)
plt.show()

cv_scores = cross_val_score(
    best_xgb,
    X_train,
    y_train,
    cv=cv_xgb,
    scoring="roc_auc"
)

print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("\n--- XGBoost Test Metrics ---")

best_xgb.fit(X_train, y_train)

y_pred = best_xgb.predict(X_test)
y_proba = best_xgb.predict_proba(X_test)

auc_test = roc_auc_score(y_test, y_proba[:, 1])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)

plot_roc_curve(y_proba, y_test)

print(f"\nXGBoost Test Metrics:")
print(f"AUC: {auc_test:.3f}")
print(f"accuracy: {accuracy:.3f}")
print(f"sensitivity: {sensitivity:.3f}")
print(f"specificity: {specificity:.3f}")

# Store results
results_summary.append({
    'Model': 'XGBoost',
    'AUC': auc_test,
    'Accuracy': accuracy,
    'Sensitivity': sensitivity,
    'Specificity': specificity
})
#%% Summary Table
print("\n" + "="*70)
print("FINAL MODEL COMPARISON")
print("="*70)

# Create summary dataframe from collected results
summary_df = pd.DataFrame(results_summary)

# Round numeric columns to 2 decimals
summary_df = summary_df.round(2)

# Print table
print("\n")
print(summary_df.to_string(index=False))

# Plot table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(cellText=summary_df.values, 
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.13, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff')

plt.title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("="*70)
# %%
