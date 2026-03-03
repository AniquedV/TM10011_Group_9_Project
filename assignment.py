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
data_missing_values = data.replace(' ', np.nan)
missing_values = data[data.isnull().any(axis=1)]

if not missing_values.empty:
    print(f"Missing values; {missing_values}")
else:
    print("No missing values")

#%%

a=3


