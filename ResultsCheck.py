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
    """y_std = np.std(y_true)
    total = 0
    for i in range(len(y_true)):
        y_true[i] = float(y_true[i])
        y_pred[i] = float(y_pred[i])
        total+=abs(y_true[i]-y_pred[i])/y_std"""

    #return np.sqrt(mean_squared_error(y_true, y_pred))
    #return total

    mu = 10
    response = 0.4
    total=0
    for i in range(len(y_true)):
        total+=abs(np.ceil((response * y_true[i]) / (response * mu - 1)) - np.ceil((response * y_pred[i]) / (response * mu - 1)))
    return total

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
            naiveResults.append(float(data[0]))
            arResults.append(float(data[1]))
            armaResults.append(float(data[2]))
            arimaResults.append(float(data[3]))
            etsResults.append(float(data[4]))
            realResults.append(float(data[6][:-1]))

esnResults = []
with open(predictionPath, 'r') as f:
    for line in f:
        esnResults.append(float(line))

esnResults_fixed = []
for i in range(len(esnResults)):
    esnResults_fixed.append(esnResults[len(esnResults)-1-i])

naiveResults = naiveResults[len(naiveResults)-len(esnResults):]
arResults = arResults[len(arResults)-len(esnResults):]
armaResults = armaResults[len(armaResults)-len(esnResults):]
arimaResults = arimaResults[len(arimaResults)-len(esnResults):]
etsResults = etsResults[len(etsResults)-len(esnResults):]
realResults = realResults[len(realResults)-len(esnResults):]

print('len esn:',len(esnResults_fixed))

print('naive:',RMSE(realResults,naiveResults))
print('ar:',RMSE(realResults,arResults))
print('arma:',RMSE(realResults,armaResults))
print('arima:',RMSE(realResults,arimaResults))
print('ets:',RMSE(realResults,etsResults))
print('esn:',RMSE(realResults,esnResults_fixed))

overload_naive = 0
underload_naive = 0
overload_ar = 0
underload_ar = 0
overload_arma = 0
underload_arma = 0
overload_arima = 0
underload_arima = 0
overload_ets = 0
underload_ets = 0
overload_esn = 0
underload_esn = 0


for i in range(len(realResults)):
    if (naiveResults[i]>=realResults[i]*1.1):
        overload_naive+=1
    elif (naiveResults[i]<=realResults[i]*0.9):
        underload_naive+=1
    if (arResults[i]>=realResults[i]*1.1):
        overload_ar+=1
    elif (arResults[i]<=realResults[i]*0.9):
        underload_ar+=1
    if (armaResults[i]>=realResults[i]*1.1):
        overload_arma+=1
    elif (armaResults[i]<=realResults[i]*0.9):
        underload_arma+=1
    if (arimaResults[i]>=realResults[i]*1.1):
        overload_arima+=1
    elif (arimaResults[i]<=realResults[i]*0.9):
        underload_arima+=1
    if (etsResults[i]>=realResults[i]*1.1):
        overload_ets+=1
    elif (etsResults[i]<=realResults[i]*0.9):
        underload_ets+=1
    if (esnResults[i]>=realResults[i]*1.1):
        overload_esn+=1
    elif (esnResults[i]<=realResults[i]*0.9):
        underload_esn+=1

print('overload of naive,ar,arma,arima,ets,esn:',overload_naive,overload_ar,overload_arma,overload_arima,overload_ets,overload_esn)
print('underload of naive,ar,arma,arima,ets,esn:',underload_naive,underload_ar,underload_arma,underload_arima,underload_ets,underload_esn)
print('total over-under of naive,ar,arma,arima,ets,esn:',overload_naive+underload_naive,overload_ar+underload_ar,overload_arma+underload_arma,
      overload_arima+underload_arima,overload_ets+underload_ets,overload_esn+underload_esn)