# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!

#%%
import numpy as np

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

print(data)


# %%

duplicates = data[data.duplicated(keep=False)]
data = data.drop_duplicates(keep='first')
print(f"Number of rows after checking for duplicates: {len(data)}")

#%%
"""Checking for missing values"""
data_missing_values = data.replace('0.0', np.nan)
missing_values = data[data.isnull().any(axis=1)]

if not missing_values.empty:
    print(f"Missing values; {missing_values}")
else:
    print("No missing values")

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

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import seaborn



data['label_bin'] = LabelEncoder().fit_transform(data['label'])
print(data[['label', 'label_bin']].head())

X = data.drop(columns=['label', 'label_bin'])
y = data['label_bin']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42        
)

# Validatie Train/test split
X_train_validatie, X_test_validatie, y_train_validatie, y_test_validatie = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42        
)

#%%

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_validatie)
X_test_scaled = scaler.transform(X_test_validatie)


scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
#%%


# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
# from sklearn.metrics import classification_report, roc_auc_score
# from scipy.stats import randint, uniform
# import sklearn.model_selection as model_selection
# from sklearn.metrics import roc_auc_score
# import sklearn.metrics as metrics

# # 2. Basismodel
# xgb = XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',   # voorkomt waarschuwingen
#     random_state=42
# )

# # 3. Zoekruimte voor hyperparameters
# param_dist = {
#     "n_estimators": randint(50, 400),
#     "max_depth": randint(2, 10),
#     "learning_rate": uniform(0.01, 0.29),   # tussen 0.01 en 0.30
#     "subsample": uniform(0.6, 0.4),         # tussen 0.6 en 1.0
#     "colsample_bytree": uniform(0.6, 0.4),  # tussen 0.6 en 1.0
#     "min_child_weight": randint(1, 10),
#     "gamma": uniform(0, 5),
#     "reg_lambda": uniform(0, 5)
# }

# # 4. Cross-validatie
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # 5. Randomized search
# search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_dist,
#     n_iter=50,              # aantal combinaties om te proberen
#     scoring='roc_auc',      # of 'f1', afhankelijk van je probleem
#     cv=cv,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42,
#     refit=True
# )

# # 6. Trainen
# search.fit(X_train_validatie, y_train_validatie)

# # 7. Beste parameters
# print("Beste parameters:")
# print(search.best_params_)

# print("\nBeste CV-score:")
# print(search.best_score_)

# # 8. Evaluatie op testset
# best_model = search.best_estimator_
# y_pred = best_model.predict(X_test_validatie)
# y_prob = best_model.predict_proba(X_test_validatie)[:, 1]

# print("\nROC AUC op testset:")
# print(roc_auc_score(y_test_validatie, y_prob))

# print("\nClassification report:")
# print(classification_report(y_test_validatie, y_pred))


#%%
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
X2 = scaler.fit_transform(X)
y2 = y.values
# Create a 20 fold stratified CV iterator
cv_20fold = model_selection.StratifiedKFold(n_splits=10)
results = []
best_n_neighbors = []

from sklearn import neighbors

train_scores = []
test_scores = []
k_list = list(range(1, 25, 2))

for k in k_list:
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(X2_train, y2_train)

    # Test the classifier on the training data and plot
    score_train = clf_knn.score(X2_train, y2_train)
    score_test = clf_knn.score(X2_test, y2_test)

    train_scores.append(score_train)
    test_scores.append(score_test)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.grid()
ax.plot(k_list, train_scores, 'o-', color="r",
        label="Training score")
ax.plot(k_list, test_scores, 'o-', color="g",
        label="Test score")

# Loop over the folds
for validation_index, test_index in cv_20fold.split(X2, y2):
    # Split the data properly
    X_validation = X2[validation_index]
    y_validation = y2[validation_index]

    X_test = X2[test_index]
    y_test = y2[test_index]

    # Create a grid search to find the optimal k using a gridsearch and 10-fold cross validation
    # Same as above
    parameters = {"n_neighbors": list(range(1, 26, 2))}
    knn = neighbors.KNeighborsClassifier()
    cv_10fold = model_selection.StratifiedKFold(n_splits=10)
    grid_search = model_selection.GridSearchCV(knn, parameters, cv=cv_10fold, scoring='roc_auc')
    grid_search.fit(X_validation, y_validation)

    # Get resulting classifier
    clf = grid_search.best_estimator_
    print(f'Best classifier: k={clf.n_neighbors}')
    best_n_neighbors.append(clf.n_neighbors)

    # Test the classifier on the test data
    probabilities = clf.predict_proba(X_test)
    scores = probabilities[:, 1]

    # Get the auc
    auc = metrics.roc_auc_score(y_test, scores)
    results.append({
        'auc': auc,
        'k': clf.n_neighbors,
        'set': 'test'
    })

    # Test the classifier on the validation data
    probabilities_validation = clf.predict_proba(X_validation)
    scores_validation = probabilities_validation[:, 1]

    # Get the auc
    auc_validation = metrics.roc_auc_score(y_validation, scores_validation)
    results.append({
        'auc': auc_validation,
        'k': clf.n_neighbors,
        'set': 'validation'
    })

# Create results dataframe and plot it
results = pd.DataFrame(results)
seaborn.boxplot(y='auc', x='set', data=results)

optimal_n = int(np.median(best_n_neighbors))
print(f"The optimal N={optimal_n}")

#%%

# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# # create model instance
# bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# # fit model
# bst.fit(X_train, y_train)
# # make predictions
# preds = bst.predict(X_test)


