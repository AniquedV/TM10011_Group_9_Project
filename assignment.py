# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!

import sys
print(sys.executable)
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics

# Some functions we will use

# Construct classifiers
svmlin = SVC(kernel='linear', gamma='scale')
svmrbf = SVC(kernel='rbf', gamma='scale')
svmpoly = SVC(kernel='poly', degree=3, gamma='scale')

clsfs = [KNeighborsClassifier(), RandomForestClassifier(), svmlin, svmpoly, svmrbf]

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



data['label_bin'] = LabelEncoder().fit_transform(data['label'])
print(data[['label', 'label_bin']].head())

X = data.drop(columns=['label', 'label_bin'])
y = data['label_bin']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svmlin = SVC(kernel='linear', gamma='scale', probability=True)
svmrbf = SVC(kernel='rbf', gamma='scale', probability=True)
svmpoly = SVC(kernel='poly', degree=3, gamma='scale', probability=True)

clsfs = [
    ("KNN", KNeighborsClassifier(n_neighbors=15)),
    ("Random Forest", RandomForestClassifier()),
    ("SVM linear", svmlin),
    ("SVM poly", svmpoly),
    ("SVM rbf", svmrbf)
]

# Plot
fig = plt.figure(figsize=(8, 8 * len(clsfs)))

for i, (name, clf) in enumerate(clsfs):
    clf.fit(X_train_scaled, y_train)

    score_train = clf.score(X_train_scaled, y_train)

    ax = fig.add_subplot(len(clsfs), 1, i + 1)
    ax.set_title(f"{name} - Training accuracy: {score_train:.3f}")

    # Plot points
    ax.scatter(
        X_train_scaled[:, 0],
        X_train_scaled[:, 1],
        marker='o',
        c=y_train,
        s=25,
        edgecolor='k',
        cmap=plt.cm.Paired
    )
plt.show()

#%%
#X2 en y2 maken
X_scaled = scaler.fit_transform(X)
y_array = y.values
# Models and parameter grids
models = {
    "KNN": {
        "model": neighbors.KNeighborsClassifier(),
        "params": {"n_neighbors": list(range(1, 26, 2))}
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10]
        }
    }
}

cv_outer = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

# Outer CV loop
for train_index, test_index in cv_outer.split(X_scaled, y_array):
    X_cv_train, X_cv_test = X_scaled[train_index], X_scaled[test_index]
    y_cv_train, y_cv_test = y_array[train_index], y_array[test_index]

    # Loop over models
    for model_name, config in models.items():
        model = config["model"]
        param_grid = config["params"]

        # Inner CV (for hyperparameter tuning)
        cv_inner = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        grid_search = model_selection.GridSearchCV(
            model,
            param_grid,
            cv=cv_inner,
            scoring='roc_auc'
        )

        grid_search.fit(X_cv_train, y_cv_train)

        best_model = grid_search.best_estimator_

        print(f"{model_name} best params: {grid_search.best_params_}")

        # Test performance
        y_probs = best_model.predict_proba(X_cv_test)[:, 1]
        auc_test = metrics.roc_auc_score(y_cv_test, y_probs)

        results.append({
            'model': model_name,
            'auc': auc_test,
            'set': 'test'
        })

        # Train performance
        y_train_probs = best_model.predict_proba(X_cv_train)[:, 1]
        auc_train = metrics.roc_auc_score(y_cv_train, y_train_probs)

        results.append({
            'model': model_name,
            'auc': auc_train,
            'set': 'train'
        })

    
# Create results dataframe and plot it
results = pd.DataFrame(results)
sns.boxplot(x='model', y='auc', hue='set', data=results)
plt.show()

#%%
if model_name == "Random Forest":
    print("Test labels:", y_cv_test)
    print("Predicted probabilities:", y_probs)
# %%
