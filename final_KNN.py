import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest, skew, kurtosis
from sklearn import neighbors
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics
import seaborn
from sklearn.metrics import roc_curve, auc, accuracy_score

from sklearn import decomposition
from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm


#%%


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

#%%

data['label_bin'] = LabelEncoder().fit_transform(data['label'])
print(data[['label', 'label_bin']].head())

X = data.drop(columns=['label', 'label_bin'])
y = data['label_bin']

X_zeros = X.replace(0.0, np.nan)  

col_missing_counts = (X == 0.0).sum()
col_missing_counts_sorted = col_missing_counts.sort_values(ascending=False)
print(col_missing_counts_sorted)

# HEATMAP PLOTTEN
plt.figure(figsize=(12,6))
seaborn.heatmap(X_zeros.isnull(), cbar=False)
# seaborn.heatmap(X_zeros.isnull(), cbar=False, xticklabels=False, yticklabels=False)
plt.title("Missing data heatmap")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()



#%%

def preprocess_data(X):
    X = X.copy()
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
    X = X.replace(outliers, np.nan)

    X = X.fillna(X.median())

    return X

X_cleaned = preprocess_data(X)



#%%

scaler = RobustScaler()
X2 = scaler.fit_transform(X_cleaned)
y2 = y.values

#%%

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)
#%%
# Create the RFE object and compute a cross-validated score.
svc = svm.SVC(kernel="linear")

# classifications
rfecv = feature_selection.RFECV(
    estimator=svc, 
    step=1,
    cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc')
rfecv.fit(X2, y2)

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


# Perform a PCA
pca = decomposition.PCA(n_components=best_n_features)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

#%%

param_grid = {"n_neighbors": list(range(1,26,2))}
k_list = list(range(1, 26, 2))

best_n_neighbors = []
all_train = []
all_test = []

# Outer CV
cv_20fold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in cv_20fold.split(X_train, y_train):

    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]

    knn = neighbors.KNeighborsClassifier()
    cv_10fold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = model_selection.GridSearchCV(
        knn,
        param_grid,
        cv=cv_10fold,
        scoring='roc_auc',
        return_train_score=True
    )

    grid_search.fit(X_cv_train, y_cv_train)

    # Beste model (zoals jij had)
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

fig = plt.figure(figsize=(8,8))
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
ax.set_ylabel("AUC")
ax.set_title("KNN learning curve (GridSearchCV)")
ax.legend()

plt.show()

print(f"The optimal N = {optimal_n}")

#%%


#%%
#KNN final classifier
# Train KNN classifier op geschaalde data
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=optimal_n)
clf_knn.fit(X_train, y_train)

score_train = clf_knn.score(X_train, y_train)
score_test = clf_knn.score(X_test, y_test)

y_pred_proba = clf_knn.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve
plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KNN Classification')
plt.legend()
plt.show()


print(f"Training accuracy: {score_train:.3f}")
print(f"Test accuracy: {score_test:.3f}")

