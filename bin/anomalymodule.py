import numpy as np
import pandas as pd
from sklearn.externals import joblib
import datetime

np.seterr(divide='ignore', invalid='ignore')

classifier = joblib.load("anomaly.sav")
featureminmax = joblib.load("featureminmax.sav")

filename = '/home/adityauj/module/bin/data/daily/'

now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")

filename = filename+date+'_Flow.csv'

dataset = pd.read_csv(filename, low_memory=False, skipinitialspace=True)
dataset = dataset.drop(['Src Port', 'Protocol', 'Label', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Dst Port', 'TotLen Fwd Pkts', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Min', 'Flow Byts/s', 'Flow IAT Mean', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Bwd Pkts/s', 'Pkt Len Min', 'FIN Flag Cnt', 'SYN Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Down/Up Ratio', 'Fwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Std'], axis=1)

featureminmax = featureminmax.convert_objects(convert_numeric=True)

for col in list(dataset.columns[:-1]):
    
        min_c, max_c = featureminmax.loc[col].values

        n_min=-25
        n_max=25
    
        dataset[col]= (((dataset[col].values - min_c)/(max_c - min_c))*(n_max - n_min) + n_min)
  
dataset.replace({np.nan:'0'}, inplace=True)
current_length = Tot_length = len(dataset)

print(classifier.predict(dataset))

while(True):
    
    dataset = pd.read_csv(filename, low_memory=False, skipinitialspace=True)
    dataset = dataset.drop(['Src Port', 'Protocol', 'Label', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Dst Port', 'TotLen Fwd Pkts', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Min', 'Flow Byts/s', 'Flow IAT Mean', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Bwd Pkts/s', 'Pkt Len Min', 'FIN Flag Cnt', 'SYN Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Down/Up Ratio', 'Fwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Std'], axis=1)
    Tot_length = len(dataset)
    
    featureminmax = featureminmax.convert_objects(convert_numeric=True)
  
    if Tot_length != current_length:
        for col in list(dataset.columns[:-1]):
        
            min_c, max_c = featureminmax.loc[col].values
    
            n_min=-25
            n_max=25
        
            dataset[col]= (((dataset[col].values - min_c)/(max_c - min_c))*(n_max - n_min) + n_min)
        
        dataset.replace({np.nan:'0'}, inplace=True)
        print(classifier.predict(dataset[current_length:]))
        
    current_length=Tot_length
                    