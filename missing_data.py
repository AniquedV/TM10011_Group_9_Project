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
from scipy import stats

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data['label_bin'] = LabelEncoder().fit_transform(data['label'])
# print(data[['label', 'label_bin']].head())

X = data.drop(columns=['label', 'label_bin'])
y = data['label_bin']

#Stukje over missing data

missing_data = X == 0.0

col_missing_percentage = X.isnull().mean() * 100
# print("Percentage missende waarden per kolom:\n", col_missing_percentage)

X = X.dropna(axis=1, thresh=len(X)*0.5)  # verwijder kolommen met >50% missende waarden
# print("\nDataframe na verwijderen van kolommen met meer dan 50% missende waarden:\n", X_cleaned)

columns_more_than_50_missing = col_missing_percentage[col_missing_percentage > 40].index.tolist()
print("Kolomnamen met meer dan 50% missende waarden:", columns_more_than_50_missing)

X.replace(0.0, np.nan, inplace=True)
# Loop door elke kolom en vervang missende waarden met de berekende mediaan
for column in X.columns:
    # Bereken de mediaan van de kolom zonder de missende waarden
    median_value = X[column].median()
    
    # Vervang missende waarden met de berekende mediaan
    X[column] = X[column].fillna(median_value)



col_missing_counts = (X == 0.0).sum()
row_missing_counts = (X == 0.0).sum(axis=1)

worst_row = row_missing_counts.idxmax()
print("Rij met meeste missing data:", worst_row)
# print(row_missing_counts)

worst_col = col_missing_counts.idxmax()
print("Kolom met meeste missing data:", worst_col)
# print(col_missing_counts)

# missing_positions = data[data == 0.0].stack()
# print(missing_positions)