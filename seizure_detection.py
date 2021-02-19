# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from matplotlib import pyplot
import pandas as pd 
import numpy as np
df = pd.read_csv('C:/Users/simon/Desktop/progetto EEG/ML-EEG-seizure-detection/EEG_seizure_detection.csv')

#%% per togliere la prima colonna dove ci sono gli ID dei pazienti
df.drop(df.columns[0],axis=1,inplace=True)

# %% 5 eyes open, 4 eyes closed, 3 recordering of healthy brain area, 2 recording of brain tumor area, 1 Recording of seizure activity 
    # quindi sostituisco tutti i numeri divers ida 1 con zero dato che voglio solo capire quando c'è stata una seizure
df['y'] = df['y'].replace([5,4,3,2],0)
# %% controllo per vedere se il totale degli 1 e degli 0 è uguale al num di registrazioni
x = df['y'].isin([1]).sum()
y = df['y'].isin([0]).sum()
print(x + y)
print(df.shape)
# %% doppio controllo, per vedere numero di null o na
q = df.isnull().sum().sum()
w = df.isna().sum().sum()
print(q,w)
# %%3. function that goes through the WHOLE dataframe making new x/ytrn e x/yval
#      ALWAYS EQUAL to "array_lenght"value
# 
#      Avendo un'organizzazione diversa, dove ogni RIGA è 1sec dei 23 (178 dati) di uno dei 500 pazient1(23+500 = 11500), per fare 1 decimo di s, 
#      bisogna selezionare le prime 18 colonne per ogni riga (df.iloc[0:1,0:18])

def sliding_window(array_lenght,num_data_in_dtframe,dtset):
    xtrn_no_output = dtset.drop(['y'],axis=1)
    data_in_1s = 170
    model = RandomForestClassifier(n_estimators=45)
    f1_tot = []
    xtrn_no_output = df.drop(['y'],axis=1)
    for raw in range(0,num_data_in_dtframe,1): #qua seleziona 1s(una "raw" da 178dati) di ogni paziente per volta
        array_start = 0
        for x in range(array_lenght,data_in_1s,array_lenght): #per ogni raw si fanno x/ytrn e x/yval della lunghezza di "array_lenght"
            xtrn = xtrn_no_output.iloc[(raw):(raw+1),(array_start):(x)] 
            ytrn_s = df.iloc[(raw):(raw+1),-1]
            ytrn = ytrn_s.to_numpy()
            xval = xtrn_no_output.iloc[(raw):(raw+1),(x):((x)+array_lenght)] 
            yval = ytrn
            model.fit(xtrn,ytrn)
            prev = model.predict(xval)
            f1_value = f1_score(yval,prev,zero_division=1)
            f1_tot.append(f1_value)
            array_start = x
    accuracy = np.mean(f1_tot)
    return accuracy

#test:                                               
#sliding_window(17,11499,df)
# %% prova ridimensionata per funzione sliding window
xtrn_no_output = df.drop(['y'],axis=1)
model = RandomForestClassifier(n_estimators=45)
f1_tot = []
for raw in range(0,2,1): #il 2 diventerà 11499
    array_start = 0
    for x in range(18,100,18): #il problema è qua
        xtrn = xtrn_no_output.iloc[(raw):(raw+1),(array_start):(x)] 
        xval = xtrn_no_output.iloc[(raw):(raw+1),(x):((x)+18)] 
        ytrn = df.iloc[(raw):(raw+1),-1]
        yval = ytrn
        model.fit(xtrn,ytrn)
        array_start = x
        prev = model.predict(xval)        
        f1_value = f1_score(yval,prev,zero_division=1)
        f1_tot.append(f1_value)
accuracy = np.mean(f1_tot)
print(f1_tot)

    
    
 #%%
 #test:                                               
mmm = sliding_window(17,100,df)   
print(mmm)    
    
    
    
    
    
    
    
    
    
    
    
    