import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Gtk3Agg')
#import matplotlib.pyplot as plt

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

def checkResult(times):
    dataPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/cran_08_10'
    #dataPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar_10_12'
    #dataPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/kyoto_10_12'

    predictionPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions/predictions_cran_historical_enet_identity_' + str(times)
    #predictionPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions_backup/edgar_10_12_components_5/predictions_edgar_historical_enet_identity_' + str(times)
    #predictionPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions_backup/kyoto_10_12_components_5/predictions_kyoto_historical_enet_identity_' + str(times)
    #predictionPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions/predictions_cran_historical_enet_identity_' + str(times)

    #predictionPath = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/cran_10_12/Ensemble_GA_cran_10_12_' + str(times)
    #predictionPath = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/edgar_10_12/Ensemble_GA_edgar_10_12_' + str(times)
    #predictionPath = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/kyoto_10_12/Ensemble_GA_kyoto_10_12_' + str(times)

    # /minh/PycharmProjects/Ensemble/PythonESN/predictions/predictions_edgar_historical_enet_identity'

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
            data = line.split(',')
            if data[0]=='Observation':
                continue
            esnResults.append(float(line))
            #esnResults.append(float(data[1]))

    esnResults_fixed = []
    for i in range(len(esnResults)):
        esnResults_fixed.append(esnResults[len(esnResults)-1-i])
    esnResults_fixed = esnResults

    """naiveResults = naiveResults[:-100]
    arResults = arResults[:-100]
    armaResults = arResults[:-100]
    arimaResults = arResults[:-100]
    etsResults = arResults[:-100]"""

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

    esn_resources = []
    arima_resources = []
    differences = []
    x_axis = []
    mu = 10
    response = 0.4
    for i in range(len(esnResults_fixed)):
        esn_resources.append(abs(np.ceil((response * realResults[i]) / (response * mu - 1)) - np.ceil((response * esnResults_fixed[i]) / (response * mu - 1))))
        arima_resources.append(abs(np.ceil((response * realResults[i]) / (response * mu - 1)) - np.ceil((response * arimaResults[i]) / (response * mu - 1))))
        differences.append(esn_resources[i]-arima_resources[i])
        x_axis.append(i)

    total_diff = []
    cur_sum = 0
    for i in range(len(differences)):
        cur_sum+=differences[i]
        total_diff.append(cur_sum)

    print('diff:',differences)
    print('cur_sum:',cur_sum)
    return cur_sum
    #plt.plot(x_axis,total_diff)
    #plt.show()

total = 0
count=5
for i in range(1,count):
    total += checkResult(i)
    print('')

print('average:',total/(count-1))