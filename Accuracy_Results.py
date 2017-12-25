import argparse
import csv
import math

import numpy as np
from sklearn.metrics import mean_squared_error

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('top_dir', type=str, help='Path to project directory')
args = parser.parse_args()

finalCRAN = args.top_dir + '/final_results/cran_10_12.csv'
finalEDGAR = args.top_dir + '/final_results/edgar_10_12.csv'
finalKyoto = args.top_dir + '/final_results/kyoto_10_12.csv'

outCRAN = args.top_dir + '/final_results/cran_accuracy_10_12.csv'
outEDGAR = args.top_dir + '/final_results/edgar_accuracy_10_12.csv'
outKyoto = args.top_dir + '/final_results/kyoto_accuracy_10_12.csv'

outList = [outCRAN, outEDGAR, outKyoto]
finalList = [finalCRAN, finalEDGAR, finalKyoto]


def responseTime(c, lamda, mu):
    p = lamda / (c * mu)
    if p >= 1:
        return -1
    r = 1 / mu + (p ** (math.sqrt(2 * c + 2))) / (p * c * (1 - p) * mu)
    return r


def resourcesCalculation(lamda, mu, r0):
    c = math.ceil(lamda / mu)
    while True:
        r = responseTime(c, lamda, mu)
        if r < 0:
            c += 1
            continue
        if (r <= r0):
            return c
        if (c >= 1000):
            print('no. of servers reached 1000')
            break
        c += 1
    return None


def mae(predictList, realList):
    results = 0
    for i in range(len(realList)):
        results += abs(float(predictList[i]) - float(realList[i]))
    return (results / len(realList))


def mape(predictList, realList):
    results = 0
    for i in range(len(realList)):
        results += abs(float(predictList[i]) - float(realList[i])) / float(realList[i]) * 100
    return (results / len(realList))


def rmse(predictList, realList):
    for i in range(len(predictList)):
        predictList[i] = float(predictList[i])
        realList[i] = float(realList[i])
    return np.sqrt(mean_squared_error(predictList, realList))


def nrmse(predictList, realList):
    for i in range(len(predictList)):
        predictList[i] = float(predictList[i])
        realList[i] = float(realList[i])
    return np.sqrt(mean_squared_error(predictList, realList)) / np.std(realList)


def doMAE(components, reals):
    compMAE = []
    for i in range(len(components)):
        compMAE.append(mae(components[i], reals))
    return compMAE


def doMAPE(components, reals):
    compMAPE = []
    for i in range(len(components)):
        compMAPE.append(mape(components[i], reals))
    return compMAPE


def doRMSE(components, reals):
    compRMSE = []
    for i in range(len(components)):
        compRMSE.append(rmse(components[i], reals))
    return compRMSE


def doNRMSE(components, reals):
    compNRMSE = []
    for i in range(len(components)):
        compNRMSE.append(nrmse(components[i], reals))
    return compNRMSE


# result: under-provision, over-provision, total wasted
def doResources(components, reals, mu, r0):
    under = []
    over = []
    total = []
    for i in range(len(components)):
        underRes = 0
        overRes = 0
        totalRes = 0
        for j in range(len(components[i])):
            if resourcesCalculation(components[i][j], mu, r0) < resourcesCalculation(reals[j], mu, r0):
                underRes += resourcesCalculation(reals[j], mu, r0) - resourcesCalculation(components[i][j], mu, r0)
                totalRes += resourcesCalculation(reals[j], mu, r0) - resourcesCalculation(components[i][j], mu, r0)
            else:
                overRes += resourcesCalculation(components[i][j], mu, r0) - resourcesCalculation(reals[j], mu, r0)
                totalRes += resourcesCalculation(components[i][j], mu, r0) - resourcesCalculation(reals[j], mu, r0)
        # yearlyCost = total/len(reals)*365
        under.append(underRes)
        over.append(overRes)
        total.append(totalRes)
    return under, over, total


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
    return names, reals


def doComponents(dataFile):
    components = []
    with open(dataFile, 'r') as f:
        count = 0
        reader = csv.reader(f)
        for line in reader:
            count += 1
            if count == 1:
                compLength = len(line)
                for i in range(compLength - 1):
                    aList = []
                    components.append(aList)
            if count > 1:
                for j in range(1, compLength):
                    components[j - 1].append(line[j])
    return components


def main():
    for i in range(len(finalList)):
        names, reals = doReals(finalList[i])
        names.insert(0, '')
        components = doComponents(finalList[i])
        compMAE = doMAE(components, reals)
        compMAE.insert(0, 'MAE')
        compMAPE = doMAPE(components, reals)
        compMAPE.insert(0, 'MAPE')
        compRMSE = doRMSE(components, reals)
        compRMSE.insert(0, 'RMSE')
        compNRMSE = doNRMSE(components, reals)
        compNRMSE.insert(0, 'NRMSE')
        under1, over1, total1 = doResources(components, reals, 10, 0.4)
        under1.insert(0, 'under - 1st scenario')
        over1.insert(0, 'over - 1st scenario')
        total1.insert(0, 'total - 1st scenario')
        under2, over2, total2 = doResources(components, reals, 10, 0.4)
        under2.insert(0, 'under - 2nd scenario')
        over2.insert(0, 'over - 2nd scenario')
        total2.insert(0, 'total - 2nd scenario')
        with open(outList[i], 'w') as f:
            writer = csv.writer(f)
            writer.writerows([names])
            writer.writerows([compMAE])
            writer.writerows([compMAPE])
            writer.writerows([compRMSE])
            writer.writerows([compNRMSE])
            writer.writerows([under1])
            writer.writerows([over1])
            writer.writerows([total1])
            writer.writerows([under2])
            writer.writerows([over2])
            writer.writerows([total2])
        print('names:', names)


if __name__ == "__main__":
    main()
