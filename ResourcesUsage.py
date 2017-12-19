import math
from os import listdir
import numpy as np

dataCRAN = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/cran_10_12'
dataEDGAR = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar_10_12'
dataKyoto = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/kyoto_10_12'

esnCRAN = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions_backup/cran_10_12_enet_identity_mae'
esnEDGAR = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions_backup/edgar_10_12_enet_identity_mae'
esnKyoto = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions_backup/kyoto_10_12_enet_identity_mae'

averageEDGAR = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/edgar_10_12/Ensemble_Average_edgar_10_12_1'

gaCRAN = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/cran_10_12'
gaEDGAR = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/edgar_10_12'
gaKyoto = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/kyoto_10_12'

gaCRANPath = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/cran_10_12/Ensemble_GA_cran_10_12_'
gaEDGARPath = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/edgar_10_12/Ensemble_GA_edgar_10_12_'
gaKyotoPath = '/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/kyoto_10_12/Ensemble_GA_kyoto_10_12_'

pathList = [gaCRANPath,gaEDGARPath,gaKyotoPath]
dataList = [dataCRAN,dataEDGAR,dataKyoto]
esnList = [esnCRAN,esnEDGAR,esnKyoto]
gaList = [gaCRAN,gaEDGAR,gaKyoto]

mu = 10
r0 = 0.4
testingSize = 240
repeatSize = 5

"""
c: number of servers
lamda: arrival rate of requests
mu: processing rate
r: response rate
p: system utilization
"""

"""def responseTime(c,lamda,mu):
    p = lamda/(c*mu)
    sum = 0
    for k in range(c):
        sum += ((c*p)**k)/math.factorial(k)
    requestProbability = 1/(1+(((1-p)*math.factorial(c)/((c*p)**c))*sum))
    if (c*mu-lamda)<=0:
        return -1
    r = requestProbability/(c*mu-lamda) + 1/mu
    return r"""

def responseTime(c,lamda,mu):
    p = lamda / (c * mu)
    if p>=1:
        return -1
    r = 1/mu + (p**(math.sqrt(2*c+2)))/(p*c*(1-p)*mu)
    return r

def resourcesCalculation(lamda,mu,r0):
    c = math.ceil(lamda/mu)
    while True:
        r = responseTime(c,lamda,mu)
        if r<0:
            c+=1
            continue
        if (r<=r0):
            return c
        if (c>=1000):
            print('no. of servers reached 1000')
            break
        c+=1
    return None

def readDatas(dataFiles,column,ignores):
    datas = []
    for i in range(testingSize):
        datas.append(0)
    for dataFile in dataFiles:
        count = 0
        with open(dataFile, 'r') as f:
            for line in f:
                count += 1
                if count > ignores:
                    data = line.split(',')
                    datas[count-ignores-1]+=float(data[column])
    for i in range(len(datas)):
        datas[i] /= len(dataFiles)
    resources = []
    for i in range(testingSize):
        # resources += resourcesCalculation(int(datas[-(i+1)]),mu,r0)
        resources.append(resourcesCalculation(int(datas[-testingSize + i]), mu, r0))
    return resources

def readData(dataFile,column,ignores):
    datas = []
    count = 0
    with open(dataFile,'r') as f:
        for line in f:
            count+=1
            if count>ignores:
                data = line.split(',')
                datas.append(float(data[column]))
    resources = []
    for i in range(testingSize):
        #resources += resourcesCalculation(int(datas[-(i+1)]),mu,r0)
        resources.append(resourcesCalculation(int(datas[-testingSize+i]),mu,r0))
    return resources

def readComponents(componentFile,dataResources):
    naiveResources = readData(componentFile,0,1)
    arResources = readData(componentFile, 1, 1)
    armaResources = readData(componentFile, 2, 1)
    arimaResources = readData(componentFile,3,1)
    etsResources = readData(componentFile, 4, 1)
    totalNaive = 0
    totalAR = 0
    totalARMA = 0
    totalARIMA = 0
    totalETS = 0
    for i in range(len(naiveResources)):
        totalNaive += abs(naiveResources[i] - dataResources[i])
        totalAR+= abs(arResources[i] - dataResources[i])
        totalARMA += abs(armaResources[i] - dataResources[i])
        totalARIMA += abs(arimaResources[i]-dataResources[i])
        totalETS += abs(etsResources[i] - dataResources[i])
    return (totalNaive,totalAR,totalARMA,totalARIMA,totalETS)

def readGA(gaDir,dataResources):
    for file in listdir(gaDir):
        resources = readData(gaDir+'/'+ file,1,1)
    list = []
    total = 0
    for i in range(len(resources)):
        total += abs(resources[i]-dataResources[i])
        list.append(abs(resources[i]-dataResources[i]))
    return sum(list)

def readGAs(gaPath, dataResources):
    finalList = []
    for i in range(1,repeatSize+1):
        resources = readData(gaPath + str(i),1,1)
        list = []
        total = 0
        for j in range(len(resources)):
            total += abs(resources[j]-dataResources[j])
            #list.append(total)
        finalList.append(total)
    return np.mean(finalList)

def readESN(esnDir,dataResources):
    listResources = []
    list = []
    sortedList = sorted(listdir(esnDir))
    ele10 = sortedList[1]
    sortedList.pop(1)
    sortedList.append(ele10)
    for i in range(len(sortedList)):
        sortedList[i]=esnDir+'/'+sortedList[i]
    #print('sortedList:',sortedList)
    total = 0
    #resources = readData(esnDir+'/'+ file,0,0)
    resources = readDatas(sortedList[:repeatSize],0,0)
    for i in range(len(resources)):
        listResources.append(abs(resources[i]-dataResources[i]))
        total += abs(resources[i]-dataResources[i])
    list.append(total)
    return list

def main():
    for i in range(len(dataList)):
        dataFile = dataList[i]
        esnDir = esnList[i]
        gaDir = gaList[i]
        dataResources = readData(dataFile,-1,1)
        #gaResources = readGA(gaDir,dataResources)
        gaResources = readGAs(pathList[i],dataResources)
        esnResources = readESN(esnDir,dataResources)
        naiveResources,arResources,armaResources,arimaResources,etsResources = readComponents(dataFile,dataResources)
        print('dataset:',dataList[i])
        #11111print('data:',dataResources)
        print('total data:',sum(dataResources))
        print('ga:',gaResources)
        print('esn:',esnResources)
        print('naive:',naiveResources)
        print('ar:',arResources)
        print('arma:',armaResources)
        print('arima:',arimaResources)
        print('ets:',etsResources)
        print()


if __name__ == "__main__":
    main()