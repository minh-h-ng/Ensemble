import numpy as np
from sklearn.metrics import mean_squared_error

def NRMSE(y_true, y_pred):
    """ Normalized Root Mean Squared Error """
    for i in range(len(y_true)):
        y_true[i] = float(y_true[i])
        y_pred[i] = float(y_pred[i])
    y_std = np.std(y_true)

    return np.sqrt(mean_squared_error(y_true, y_pred))/y_std

def RMSE(y_true, y_pred):
    """ Root Mean Squared Error """
    for i in range(len(y_true)):
        y_true[i] = float(y_true[i])
        y_pred[i] = float(y_pred[i])
    y_std = np.std(y_true)

    return np.sqrt(mean_squared_error(y_true, y_pred))

dataPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar'
predictionPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions/predictions_edgar_historical_enet_identity'

naiveResults = []
arResults = []
armaResults = []
arimaResults = []
etsResults = []
realResults = []

count = 0
with open(dataPath,'r') as f:
    for line in f:
        count+=1
        if count>1:
            data = line.split(',')
            naiveResults.append(data[0])
            arResults.append(data[1])
            armaResults.append(data[2])
            arimaResults.append(data[3])
            etsResults.append(data[4])
            realResults.append(data[6][:-1])

averageResults = []
for i in range(len(naiveResults)):
    averageResults.append((float(naiveResults[i])+float(arResults[i])+float(armaResults[i])
                           +float(arimaResults[i])+float(etsResults[i]))/5)

errorResults = []

with open(predictionPath, 'r') as f:
    for line in f:
        errorResults.append(line[:-1])

naiveResults = naiveResults[(len(naiveResults)-len(errorResults)):]
arResults = arResults[(len(arResults)-len(errorResults)):]
armaResults = armaResults[(len(armaResults)-len(errorResults)):]
arimaResults = arimaResults[(len(arimaResults)-len(errorResults)):]
etsResults = etsResults[(len(etsResults)-len(errorResults)):]
realResults = realResults[(len(realResults)-len(errorResults)):]

esnResults = []
for i in range(len(errorResults)):
    esnResults.append(float(errorResults[i])+averageResults[len(averageResults)-len(errorResults)+i])

print('naive:',RMSE(realResults,naiveResults))
print('ar:',RMSE(realResults,arResults))
print('arma:',RMSE(realResults,armaResults))
print('arima:',RMSE(realResults,arimaResults))
print('ets:',RMSE(realResults,etsResults))
print('esn:',RMSE(realResults,esnResults))