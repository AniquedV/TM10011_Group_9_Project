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

# col_index1 = data.columns.get_loc("tf_GLRLM_ZoneVariance")
# col_index2 = data.columns.get_loc("tf_GLRLM_GrayLevelVariance")
#%% Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest, skew, kurtosis
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics
from scipy import stats



#%% visualisatie

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data['label_bin'] = LabelEncoder().fit_transform(data['label'])
# print(data[['label', 'label_bin']].head())

X = data.drop(columns=['label', 'label_bin'])
y = data['label_bin']

#%%
cols_with_missing = [col for col in X.columns if (X[col] == 0.0).any()]

for col in cols_with_missing:
    missing_indicator = (X[col]==0).astype(int)
    missing_by_label = pd.DataFrame({
        'label': y,
        'missing': missing_indicator
    })
    
X_zeros = X.replace(0.0, np.nan)    

# # HEATMAP PLOTTEN
# plt.figure(figsize=(12,6))
# seaborn.heatmap(X_zeros.isnull(), cbar=False)
# # seaborn.heatmap(X_zeros.isnull(), cbar=False, xticklabels=False, yticklabels=False)
# plt.title("Missing data heatmap")
# plt.xlabel("Features")
# plt.ylabel("Samples")
# plt.show()

#%%
# def preprocess_data(X):
#     X = X.copy()

#     Q1 = X.quantile(0.25)
#     Q3 = X.quantile(0.75)
#     IQR = Q3 - Q1

#     outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
#     X = X.mask(outliers, np.nan)

#     X = X.fillna(X.median())

#     return X

# X = preprocess_data(X)


    plot_data = missing_by_label.groupby('label')['missing'].mean()
    
    plot_data.plot(kind='bar')
    plt.ylabel('Fractie missing')
    plt.title(f'Missingness van {col} per label')
    plt.ylim(0, 1)
    plt.show()


# import missingno as msno

# msno.matrix(X)
# plt.show()

# def detect_missing_data(X):
#     X = X.copy()
#     X = X.replace(0.0, np.nan)
#     return X


# def detect_outliers(X):
#     detect_missing_data(X)
#     Q1 = X.quantile(0.25)
#     Q3 = X.quantile(0.75)
#     IQR = Q3 - Q1

#     outliers_iqr = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))
#     X = X.replace(outliers_iqr, np.nan)
#     return X, IQR, outliers_iqr

# def imputation(X):
#     detect_missing_data(X)
#     detect_outliers(X)
#     for column in X.columns:
#         median_value = X[column].median()
        
#         X[column] = X[column].fillna(median_value)
#     return X
 

col_missing_counts = (X == 0.0).sum()
row_missing_counts = (X == 0.0).sum(axis=1)

worst_row = row_missing_counts.idxmax()
print("Rij met meeste missing data:", worst_row)
# print(row_missing_counts)

worst_col = col_missing_counts.idxmax()
print("Kolom met meeste missing data:", worst_col)

# Sorteer van groot naar klein
col_missing_counts_sorted = col_missing_counts.sort_values(ascending=False)

print(col_missing_counts_sorted)
# print(col_missing_counts)

# missing_positions = data[data == 0.0].stack()
# print(missing_positions)

#%%

# plt.figure(figsize=(12, 6))
# seaborn.boxplot(data=X)
# plt.xticks(rotation=45)
# plt.title("Boxplots of Features")
# plt.show()












