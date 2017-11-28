import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("baggingFile", type=str)
args = parser.parse_args()

datafile = args.datafile
baggingFile = args.baggingFile

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

def ensemble_bagging():
    results = []
    if (len(naiveList)==0 or len(arList)==0 or len(armaList)==0 or len(arimaList)==0 or len(etsList)==0 or len(realList)==0):
        print('Component or Real Lists Uninitialized!')
    else:
        for i in range(len(naiveList)):
            results.append(np.ceil((naiveList[i]+arList[i]+armaList[i]+arimaList[i]+etsList[i])/5))
    results = results[-342:]
    with open(baggingFile,'w') as f:
        for i in range(len(results)):
            f.writelines(str(results[i])+'\n')


def main():
    getAlgorithms(dataFile=datafile)
    ensemble_bagging()

if __name__ == '__main__':
    main()