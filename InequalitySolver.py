import numpy as np

"""def calculateNumberOfServers(lambd, mu, response):
    c_start_point = lambd / mu
    if (c_start_point - int(lambd/mu))==0:
        c_start_point = int(c_start_point+1)
    else:
        c_start_point= int(np.ceil(c_start_point))

    found = False
    for c in range(c_start_point, 10000):
        p = lambd / (c * mu)
        if p >= 1 or p <= 0:
            print('p problem at lambda:',lambd)
        stretch_factor_form = 1 + (p ** (np.sqrt(2 * c + 2))) / (p * c * (1 - p))
        equation = 1 / mu * stretch_factor_form
        #print('equation:',equation)
        # equation = (1 / (1-p**c)) * (1/mu)
        # print('equation:',equation)
        if equation > response:
            found = True
        if found == True:
            # print('result:',(c-1))
            return (c-1)
    if found == False:
        print('cannot find for lambda:',lambd)
    return None"""

def calculateNumberOfServers(lambd, mu, response):
    formula = (response * lambd)/(response*mu-1)

    return np.ceil(formula)

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

realResources = []
naiveResources = []
arResources = []
armaResources = []
arimaResources = []
etsResources = []
esnResources = []

mu = 10
response = 0.4

for i in range(len(realResults)):
    lambda_real = realResults[i]
    lambda_naive = naiveResults[i]
    lambda_ar = arResults[i]
    lambda_arma = armaResults[i]
    #lambda_arima = arimaResults[i]
    lambda_ets = etsResults[i]
    lambda_esn = esnResults_fixed[i]
    realResources.append(calculateNumberOfServers(lambda_real,mu,response))
    naiveResources.append(calculateNumberOfServers(lambda_naive, mu, response))
    arResources.append(calculateNumberOfServers(lambda_ar, mu, response))
    armaResources.append(calculateNumberOfServers(lambda_arma, mu, response))
    #arimaResources.append(calculateNumberOfServers(lambda_arima, mu, response))
    etsResources.append(calculateNumberOfServers(lambda_ets, mu, response))
    esnResources.append(calculateNumberOfServers(lambda_esn, mu, response))

total_naive = 0
total_ar = 0
total_arma = 0
total_arima = 0
total_ets = 0
total_esn = 0
for i in range(len(realResources)):
    total_naive += abs(naiveResources[i]-realResources[i])
    total_ar += abs(arResources[i] - realResources[i])
    total_arma += abs(armaResources[i] - realResources[i])
    #total_arima += abs(arimaResources[i] - realResources[i])
    total_ets += abs(etsResources[i]-realResources[i])
    total_esn += abs(esnResources[i]-realResources[i])

print('total of naive,ar,arma,arima,ets,esn:',total_naive,total_ar,total_arma,total_arima,total_ets,total_esn)
