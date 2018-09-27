import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

dataset = pd.read_csv("sample.csv",low_memory=False,skipinitialspace=True)
dataset = dataset.loc[:,~dataset.columns.str.replace("(\.\d+)$","").duplicated()]
dataset.iloc[:,-1] = dataset.iloc[:,-1].astype('category')

cat_columns = dataset.select_dtypes(['category']).columns

dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)

dataset.replace({'Infinity': '0', np.nan:'0'}, inplace=True)

dataset['Flow Bytes/s'] = dataset['Flow Bytes/s'].str.replace('','').astype(np.float64)
dataset['Flow Packets/s'] = dataset['Flow Packets/s'].str.replace('','').astype(np.float64)

for col in list(dataset.columns[:-1]):
    
    max_c=dataset[col].max()
    min_c=dataset[col].min()
    
    n_min=-30
    n_max=30

    dataset[col]= (((dataset[col].values - min_c)/(max_c - min_c))*(n_max - n_min) + n_min)
