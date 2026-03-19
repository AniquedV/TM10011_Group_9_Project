
# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


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
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest, skew, kurtosis
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics
import seaborn


# Some functions we will use

#wordt niet meer gebruikt
def colorplot(clf, ax, x, y, h=100):
    '''
    Overlay the decision areas as colors in an axes.

    Input:
        clf: trained classifier
        ax: axis to overlay color mesh on
        x: feature on x-axis
        y: feature on y-axis
        h(optional): steps in the mesh
    '''
    # Create a meshgrid the size of the axis
    xstep = (x.max() - x.min() ) / 20.0
    ystep = (y.max() - y.min() ) / 20.0
    x_min, x_max = x.min() - xstep, x.max() + xstep
    y_min, y_max = y.min() - ystep, y.max() + ystep
    h = max((x_max - x_min, y_max - y_min))/h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if len(Z.shape) > 1:
        Z = Z[:, 1]

    # Put the result into a color plot
    cm = plt.cm.RdBu_r
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    del xx, yy, x_min, x_max, y_min, y_max, Z, cm

#%% visualisatie
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV


data['label_bin'] = LabelEncoder().fit_transform(data['label'])
print(data[['label', 'label_bin']].head())

X = data.drop(columns=['label', 'label_bin'])
y = data['label_bin']

X_array = X.values
y_array = y.values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}

cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv_inner,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

#%% BEST MODEL (trained on train folds)
best_model = grid_search.best_estimator_


#%% CROSS-VAL PERFORMANCE OP TRAIN
cv_scores = model_selection.cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=cv_inner,
    scoring='roc_auc'
)

print(f"CV AUC (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

#%% TESTSET

y_test_probs = best_model.predict_proba(X_test)[:, 1]
auc_test = metrics.roc_auc_score(y_test, y_test_probs)

print(f"\nTEST AUC: {auc_test:.3f}")







'''
# Train KNN classifier op geschaalde data
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=15)
clf_knn.fit(X_train_scaled, y_train)

# Training accuracy
score_train = clf_knn.score(X_train_scaled, y_train)

# Plot decision boundary
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_title(f"Training performance: accuracy {score_train}")

# Correct: gebruik NumPy array van geschaalde data
#colorplot(clf_knn, ax, X_train_scaled[:, 0], X_train_scaled[:, 1], h=1000)

# Plot de punten
ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], marker='o', c=y_train,
           s=25, edgecolor='k', cmap=plt.cm.Paired)
plt.show()


#%%
#X2 en y2 maken
X_scaled = scaler.fit_transform(X)
y_array = y.values
# Create a 20 fold stratified CV iterator
cv_20fold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []
best_n_neighbors = []

param_grid = {"n_neighbors": list(range(1,26,2))}

# Loop over the folds
for train_index, test_index in cv_20fold.split(X_scaled, y_array):
    # Split the data properly
    X_cv_train, X_cv_test = X_scaled[train_index], X_scaled[test_index]
    y_cv_train, y_cv_test = y_array[train_index], y_array[test_index]

    # Create a grid search to find the optimal k using a gridsearch and 10-fold cross validation
    # Same as above
    knn = neighbors.KNeighborsClassifier()
    cv_10fold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = model_selection.GridSearchCV(knn, param_grid, cv=cv_10fold, scoring='roc_auc')
    grid_search.fit(X_cv_train, y_cv_train)

    # Get resulting classifier
    clf = grid_search.best_estimator_
    best_n_neighbors.append(clf.n_neighbors)
    print(f'best k in fold: {clf.n_neighbors}')

    # Test the classifier on the test data
    y_probs = clf.predict_proba(X_cv_test)[:,1]

    # Get the auc
    auc = metrics.roc_auc_score(y_cv_test, y_probs)
    results.append({
        'auc': auc,
        'k': clf.n_neighbors,
        'set': 'test'
    })

    # Test the classifier on the validation data
    y_train_probs = clf.predict_proba(X_cv_train)[:,1]
    auc_train = metrics.roc_auc_score(y_cv_train, y_train_probs)
    results.append({
        'auc': auc_train,
        'k': clf.n_neighbors,
        'set': 'train'
    })

# Create results dataframe and plot it
results = pd.DataFrame(results)
sns.boxplot(y='auc', x='set', data=results)

optimal_n = int(np.median(best_n_neighbors))
print(f"The optimal N={optimal_n}")
'''