import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("crossFile", type=str)
args = parser.parse_args()

datafile = args.datafile
crossFile = args.crossFile

naiveList = []
arList = []
armaList = []
arimaList = []
etsList = []
realList = []

def getAlgorithms(dataFile):
    with open(dataFile,'r') as f:
        lineCount = 0
        for line in f:
            #skip first line
            lineCount+=1
            if lineCount>1:
                line = line[:-1]
                lineParts = line.split(',')
                naiveList.append(int(float(lineParts[0])))
                arList.append(int(float(lineParts[1])))
                armaList.append(int(float(lineParts[2])))
                arimaList.append(int(float(lineParts[3])))
                etsList.append(int(float(lineParts[4])))
                realList.append(int(float(lineParts[6])))
    no_test = 342
    testSet = realList[-342:]
    trainSet = realList[:-342]
    return trainSet,testSet

def ensemble_cross_validation(trainSet, testSet, noFold):
    print('')


def main():
    trainSet, testSet = getAlgorithms(dataFile=datafile)
    print('len:',len(trainSet),len(testSet),len(trainSet)+len(testSet))

if __name__ == '__main__':
    main()