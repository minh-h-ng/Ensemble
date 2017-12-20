import csv
import numpy as np
from sklearn.metrics import mean_squared_error

finalCRAN = '/home/minh/PycharmProjects/Ensemble/final_results/cran_10_12.csv'
finalEDGAR = '/home/minh/PycharmProjects/Ensemble/final_results/edgar_10_12.csv'
finalKyoto = '/home/minh/PycharmProjects/Ensemble/final_results/kyoto_10_12.csv'

outCRAN = '/home/minh/PycharmProjects/Ensemble/final_results/cran_accuracy_10_12.csv'
outEDGAR = '/home/minh/PycharmProjects/Ensemble/final_results/edgar_accuracy_10_12.csv'
outKyoto = '/home/minh/PycharmProjects/Ensemble/final_results/kyoto_accuracy_10_12.csv'

outList = [outCRAN,outEDGAR,outKyoto]
finalList = [finalCRAN,finalEDGAR,finalKyoto]

def mae(predictList,realList):
    results = 0
    for i in range(len(realList)):
        results+=abs(float(predictList[i])-float(realList[i]))
    return (results/len(realList))

def mape(predictList,realList):
    results = 0
    for i in range(len(realList)):
        results+=abs(float(predictList[i])-float(realList[i]))/float(realList[i])*100
    return (results / len(realList))

def rmse(predictList,realList):
    for i in range(len(predictList)):
        predictList[i] = float(predictList[i])
        realList[i] = float(realList[i])
    return np.sqrt(mean_squared_error(predictList,realList))

def nrmse(predictList,realList):
    for i in range(len(predictList)):
        predictList[i] = float(predictList[i])
        realList[i] = float(realList[i])
    return np.sqrt(mean_squared_error(predictList,realList))/np.std(realList)

def doMAE(components,reals):
    compMAE = []
    for i in range(len(components)):
        compMAE.append(mae(components[i],reals))
    return compMAE

def doMAPE(components,reals):
    compMAPE = []
    for i in range(len(components)):
        compMAPE.append(mape(components[i],reals))
    return compMAPE

def doRMSE(components,reals):
    compRMSE = []
    for i in range(len(components)):
        compRMSE.append(rmse(components[i],reals))
    return compRMSE

def doNRMSE(components,reals):
    compNRMSE = []
    for i in range(len(components)):
        compNRMSE.append(nrmse(components[i],reals))
    return compNRMSE

def doReals(dataFile):
    reals = []
    with open(dataFile, 'r') as f:
        count = 0
        reader = csv.reader(f)
        for line in reader:
            count += 1
            if count == 1:
                names = line
            if count > 1:
                reals.append(line[0])
    names = names[1:]
    return names,reals

def doComponents(dataFile):
    components = []
    with open(dataFile, 'r') as f:
        count = 0
        reader = csv.reader(f)
        for line in reader:
            count+=1
            if count==1:
                compLength = len(line)
                for i in range(compLength-1):
                    aList = []
                    components.append(aList)
            if count>1:
                for j in range(1,compLength):
                    components[j-1].append(line[j])
    return components

def main():
    for i in range(len(finalList)):
        names, reals = doReals(finalList[i])
        components = doComponents(finalList[i])
        compMAE = doMAE(components,reals)
        compMAPE = doMAPE(components,reals)
        compRMSE = doRMSE(components,reals)
        compNRMSE = doNRMSE(components,reals)
        with open(outList[i],'w') as f:
            writer = csv.writer(f)
            writer.writerows([names])
            writer.writerows([compMAE])
            writer.writerows([compMAPE])
            writer.writerows([compRMSE])
            writer.writerows([compNRMSE])
        print('names:',names)

if __name__ == "__main__":
    main()