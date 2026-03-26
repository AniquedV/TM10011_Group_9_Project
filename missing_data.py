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
from scipy import stats



#%% visualisatie

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data['label_bin'] = LabelEncoder().fit_transform(data['label'])
# print(data[['label', 'label_bin']].head())

X = data.drop(columns=['label', 'label_bin'])
y = data['label_bin']

#Stukje over missing data

X = X.replace(0.0, np.nan)
col_missing_percentage = X.isnull().mean() * 100

X = X.dropna(axis=1, thresh=len(X)*0.5)  # verwijder kolommen met >50% missende waarden
# print("\nDataframe na verwijderen van kolommen met meer dan 50% missende waarden:\n", X_cleaned)

columns_more_than_50_missing = col_missing_percentage[col_missing_percentage > 50].index.tolist()
print("Kolomnamen met meer dan 50% missende waarden:", columns_more_than_50_missing)

missing_percent = X.isnull().mean() * 100

missing_percent.sort_values(ascending=False).plot(kind='bar', figsize=(12,5))
plt.ylabel("Percentage missing")
plt.title("Missing data per feature")
plt.show()

cols_with_missing = X.columns[X.isnull().any()]

for col in cols_with_missing:
    missing_indicator = X[col].isnull().astype(int)
    missing_by_label = pd.DataFrame({
        'label': y,
        'missing': missing_indicator
    })
    
    plot_data = missing_by_label.groupby('label')['missing'].mean()
    
    plot_data.plot(kind='bar')
    plt.ylabel('Fractie missing')
    plt.title(f'Missingness van {col} per label')
    plt.ylim(0, 1)
    plt.show()


# plt.figure(figsize=(12,6))
# sns.heatmap(X.isnull(), cbar=False, xticklabels=False, yticklabels=False)
# plt.title("Missing data heatmap")
# plt.xlabel("Features")
# plt.ylabel("Samples")
# plt.show()


# import missingno as msno

# msno.matrix(X)
# plt.show()


# X.replace(0.0, np.nan, inplace=True)
# # Loop door elke kolom en vervang missende waarden met de berekende mediaan
# for column in X.columns:
#     # Bereken de mediaan van de kolom zonder de missende waarden
#     median_value = X[column].median()
    
#     # Vervang missende waarden met de berekende mediaan
#     X[column] = X[column].fillna(median_value)



# col_missing_counts = (X == 0.0).sum()
# row_missing_counts = (X == 0.0).sum(axis=1)

# worst_row = row_missing_counts.idxmax()
# print("Rij met meeste missing data:", worst_row)
# # print(row_missing_counts)

# worst_col = col_missing_counts.idxmax()
# print("Kolom met meeste missing data:", worst_col)
# # print(col_missing_counts)

# # missing_positions = data[data == 0.0].stack()
# # print(missing_positions)