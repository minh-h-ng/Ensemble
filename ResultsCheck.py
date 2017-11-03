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
    total = 0
    for i in range(len(y_true)):
        y_true[i] = float(y_true[i])
        y_pred[i] = float(y_pred[i])
        total+=abs(y_true[i]-y_pred[i])

    return np.sqrt(mean_squared_error(y_true, y_pred))
    #return total

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

esnResults = []
with open(predictionPath, 'r') as f:
    for line in f:
        esnResults.append(float(line))

print('naive:',RMSE(realResults,naiveResults))
print('ar:',RMSE(realResults,arResults))
print('arma:',RMSE(realResults,armaResults))
print('arima:',RMSE(realResults,arimaResults))
print('ets:',RMSE(realResults,etsResults))
print('esn:',RMSE(realResults,esnResults))