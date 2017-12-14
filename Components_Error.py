import numpy as np
from sklearn.metrics import mean_squared_error

cranFile = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/cran_10_12'
edgarFile = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar_10_12'
kyotoFile = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/kyoto_10_12'

def doCalculate(componentList,realList,type):
    print(type)
    print('mae:',mae(componentList,realList))
    print('mape:',mape(componentList,realList))
    print('rmse:',rmse(componentList,realList))

def mae(predictList,realList):
    total = 0
    for i in range(len(realList)):
        total += abs(predictList[i]-realList[i])
    mae = total/len(realList)
    return mae

def mape(predictList,realList):
    total = 0
    for i in range(len(realList)):
        total += abs(predictList[i]-realList[i])/realList[i]
    mape = total/len(realList) * 100
    return mape

def rmse(predictList,realList):
    return np.sqrt(mean_squared_error(predictList,realList))


def doDataset(dataFile):
    realList = []
    naiveList = []
    arList = []
    armaList = []
    arimaList = []
    etsList = []
    with open(dataFile,'r') as f:
        count = 0
        for line in f:
            count += 1
            if count>1:
                linePart = line.split(',')
                naiveList.append(float(linePart[0]))
                arList.append(float(linePart[1]))
                armaList.append(float(linePart[2]))
                arimaList.append(float(linePart[3]))
                etsList.append(float(linePart[4]))
                realList.append(float(linePart[6]))
    naiveList = naiveList[-342:]
    arList = arList[-342:]
    armaList = armaList[-342:]
    arimaList = arimaList[-342:]
    etsList = etsList[-342:]
    realList = realList[-342:]
    doCalculate(naiveList,realList,'naive')
    doCalculate(arList,realList,'ar')
    doCalculate(armaList, realList, 'arma')
    doCalculate(arimaList, realList, 'arima')
    doCalculate(etsList, realList, 'ets')

def main():
    print('CRAN dataset:')
    doDataset(cranFile)
    print()
    print('EDGAR dataset:')
    doDataset(edgarFile)
    print()
    print('Kyoto dataset:')
    doDataset(kyotoFile)

if __name__ == "__main__":
    main()