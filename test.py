import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time
import datetime

np.seterr(divide='ignore', invalid='ignore')

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st)

dataset = pd.read_csv("sample.csv",low_memory=False,skipinitialspace=True)
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

X = dataset.drop('Label', axis=1)
y = dataset['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st)