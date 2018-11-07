import numpy as np
import pandas as pd
from sklearn import decomposition
import time
import datetime

np.seterr(divide='ignore', invalid='ignore')

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st)

dataset = pd.read_csv("C:/Users/Aditya Ujeniya/.spyder-py3/sample.csv",low_memory=True,skipinitialspace=True)
dataset = dataset.loc[:,~dataset.columns.str.replace("(\.\d+)$","").duplicated()]
dataset.iloc[:,-1] = dataset.iloc[:,-1].astype('category')

cat_columns = dataset.select_dtypes(['category']).columns

dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)

dataset.replace({'Infinity': '0', np.nan:'0'}, inplace=True)

dataset['Flow Bytes/s'] = dataset['Flow Bytes/s'].str.replace('','').astype(np.float64)
dataset['Flow Packets/s'] = dataset['Flow Packets/s'].str.replace('','').astype(np.float64)
index_col = []

for col in list(dataset.columns[:-1]):
    
    max_c=dataset[col].max()
    min_c=dataset[col].min()
    
    n_min=-25
    n_max=25

    dataset[col]= (((dataset[col].values - min_c)/(max_c - min_c))*(n_max - n_min) + n_min)
    if dataset[col].isnull().values.any() == True:
        dataset.drop(col, axis=1, inplace = True)
        
        
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st)

distinct_columns = set()

correlation_data = dataset.corr(method = 'pearson', min_periods=3).abs().unstack()
correlation_data = correlation_data.sort_values(ascending=False)

list_columns = []
for col_a in dict(correlation_data):
    if correlation_data[col_a] >= 0.95 and correlation_data[col_a] < 1:
        list_columns.append(col_a)

for col1, col2 in list_columns:
    distinct_columns.add(col1)
    distinct_columns.add(col2)

for cols in set(dataset.columns[:-1]):
    if(cols not in distinct_columns):
        dataset.drop(cols, axis=1, inplace = True)
        
        
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st)

print(distinct_columns)

labels = dataset[['Label']].copy()
dataset.drop(dataset.columns[-1], axis = 1, inplace = True)

pca = decomposition.PCA(n_components = 7)
pca.fit(dataset)
dataset= pca.transform(dataset)

dataset = pd.DataFrame(dataset)
dataset['Label'] = labels['Label']

dataset.to_csv("reduced.csv", sep=',');

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st)