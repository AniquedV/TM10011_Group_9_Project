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
import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

# Example: data should already exist
# data = [[y1, x1], [y2, x2], ...]

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

# Select columns
y = df.iloc[:, 0]   # first column → y-axis
x = df.iloc[:, 1]   # second column → x-axis

# Create scatter plot
plt.figure()
plt.scatter(x, y)
plt.xlabel("Second column")
plt.ylabel("First column")
plt.title("Scatter plot of column 1 vs column 2")
plt.show()