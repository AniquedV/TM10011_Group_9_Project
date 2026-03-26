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
from sklearn.metrics import roc_curve, auc

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

#%%

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

#%%

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

#%%
# Create a 20 fold stratified CV iterator
cv_20fold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []
best_n_neighbors = []

param_grid = {"n_neighbors": list(range(1,26,2))}

all_train = []
all_test = []

# Loop over the folds
for train_index, test_index in cv_20fold.split(X_train, y_train):
    
    train_scores = []
    test_scores = []
    # Split the data properly
    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]


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

    # Test the classifier on the training data and plot
    train_proba = clf.predict_proba(X_cv_train)[:, 1]
    test_proba = clf.predict_proba(X_cv_test)[:, 1]

    score_train = metrics.roc_auc_score(y_cv_train, train_proba)
    score_test = metrics.roc_auc_score(y_cv_test, test_proba)

    train_scores.append(score_train)
    test_scores.append(score_test)

    all_train.append(train_scores)
    all_test.append(test_scores)

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

#%%



# Repeat the experiment 20 times, use 20 random splits in which class balance is retained
sss = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=0)

for train_index, test_index in sss.split(X2, y2):
    train_scores = []
    test_scores = []

    split_X_train = X2[train_index]
    split_y_train = y2[train_index]
    split_X_test = X2[test_index]
    split_y_test = y2[test_index]

    for k in k_list:
        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf_knn.fit(split_X_train, split_y_train)

        # Test the classifier on the training data and plot
        train_proba = clf_knn.predict_proba(split_X_train)[:, 1]
        test_proba = clf_knn.predict_proba(split_X_test)[:, 1]

        score_train = metrics.roc_auc_score(split_y_train, train_proba)
        score_test = metrics.roc_auc_score(split_y_test, test_proba)


        train_scores.append(score_train)
        test_scores.append(score_test)

    all_train.append(train_scores)
    all_test.append(test_scores)


# Create numpy array of scores and calculate the mean and std
all_train = np.array(all_train)
all_test = np.array(all_test)

train_scores_mean = all_train.mean(axis=0)
train_scores_std = all_train.std(axis=0)

test_scores_mean = all_test.mean(axis=0)
test_scores_std = all_test.std(axis=0)

# Plot the mean scores and the std as shading
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.grid()
ax.fill_between(k_list, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
ax.fill_between(k_list, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
ax.plot(k_list, train_scores_mean, 'o-', color="r",
        label="Training score")
ax.plot(k_list, test_scores_mean, 'o-', color="g",
        label="Test score")


