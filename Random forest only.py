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
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# Functions we will use
def plot_learning_curve(estimator, title, X, y, axes, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid()
    axes.fill_between(train_sizes,
                      train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std,
                      alpha=0.1)

    axes.fill_between(train_sizes,
                      test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std,
                      alpha=0.1)

    axes.plot(train_sizes, train_scores_mean, 'o-', label="Training AUC")
    axes.plot(train_sizes, test_scores_mean, 'o-', label="Validation AUC")

    axes.legend(loc="best")

    return axes

def plot_roc_curve(y_score, y_truth):
    '''
    Plot an ROC curve.
    '''
    # Only take scores for class = 1
    y_score = y_score[:, 1]

    fpr, tpr, _ = roc_curve(y_truth, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr,
             lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1],
             lw=1, linestyle='--', label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Random Forest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Prepare data
data['label_bin'] = LabelEncoder().fit_transform(data['label'])

X = data.drop(columns=['label', 'label_bin']).values
y = data['label_bin'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# Define model
model = RandomForestClassifier(random_state=42)

# Define hyperparameter distributions for RandomizedSearch
param_distributions = {
    "n_estimators": randint(50, 200),         #number of trees
    "max_depth": randint(2, 10),              #max depth of trees
    "min_samples_split": randint(2, 20),      #min samples to split
    "min_samples_leaf": randint(1, 20),       #min samples per leaf
}

# Stratified CV
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#%% RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,                  # amount of random iterations
    scoring='roc_auc',
    cv=cv_inner,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV on training data
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

#%% Learning curve 
fig, ax = plt.subplots(figsize=(7,5))

plot_learning_curve(
    estimator=best_model,
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
#%% Final evaluation on test set
y_test_probs = best_model.predict_proba(X_test)
auc_test = roc_auc_score(y_test, y_test_probs[:, 1])

print("Final Test AUC:", auc_test)

# Performance scores
y_test_pred = best_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
accuracy = accuracy_score(y_test, y_test_pred)
sensitivity = recall_score(y_test, y_test_pred)  # same as tp / (tp+fn)
specificity = tn / (tn + fp)
print("accuracy:", accuracy)
print("sensitivity:", sensitivity)
print("specificity:", specificity)

# ROC curve
plot_roc_curve(y_test_probs, y_test)