import numpy as np
import pandas as pd
from numpy.linalg import eig
import time
import datetime

np.seterr(divide='ignore', invalid='ignore')

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print("Started preprocessing\n\n",st)

dataset = pd.read_csv("C:/Users/Aditya Ujeniya/.spyder-py3/dataset.csv",low_memory=True,skipinitialspace=True)
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
print("\nDone with preprocessing\nStarting Correlation\n\n",st)

distinct_columns = set()

correlation_data = dataset.corr(method = 'pearson', min_periods=3).abs().unstack()
correlation_data = correlation_data.sort_values(ascending=False)

list_columns = []
for col_a in dict(correlation_data):
    if correlation_data[col_a] >= 0.80 and correlation_data[col_a] < 1:
        list_columns.append(col_a)

for col1, col2 in list_columns:
    distinct_columns.add(col1)
    distinct_columns.add(col2)

for cols in set(dataset.columns[:-1]):
    if(cols not in distinct_columns):
        dataset.drop(cols, axis=1, inplace = True)
        
len_of_columns = len(distinct_columns)        

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print("\nDone with correlation\nStarting with PCA\n\n",st)

dataset.dropna(how="all", inplace=True)

X = dataset.drop('Label', axis=1)
labels = dataset[['Label']].copy()

mean_of_dataset = np.mean(X.T, axis=1)

centering_values = X - mean_of_dataset

covariance_of_dataset = np.cov(centering_values.T)

values, vectors = eig(covariance_of_dataset)

eig_pairs = [(np.abs(values[i]), vectors[:,i]) for i in range(len(values))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

matrix_w = np.hstack((eig_pairs[0][1].reshape(len_of_columns,1), eig_pairs[1][1].reshape(len_of_columns,1), eig_pairs[2][1].reshape(len_of_columns,1), eig_pairs[3][1].reshape(len_of_columns,1), eig_pairs[4][1].reshape(len_of_columns,1), eig_pairs[5][1].reshape(len_of_columns,1), eig_pairs[6][1].reshape(len_of_columns,1)))
transformed = matrix_w.T.dot(X.T)
transformed = transformed.T

dataset = transformed.real
dataset = pd.DataFrame(dataset)

dataset['Label'] = labels['Label']

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print("\nDone with PCA\nStarting to write to file\n\n",st)

dataset.to_csv("reduced.csv", sep=',');

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print("\nDone with file writing\n\n",st)