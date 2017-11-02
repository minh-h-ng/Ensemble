
naiveResults = []
arResults = []
armaResults = []
arimaResults = []
etsResults = []
realResults = []

dataPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar'
writePath = '/home/minh/PycharmProjects/Ensemble/PythonESN/data/edgar_historical'

lineCount = 0
with open(dataPath,'r') as f:
    for line in f:
        # skip the title line
        if lineCount>=1:
            data = line.split(',')
            naiveResults.append(data[0])
            arResults.append(data[1])
            armaResults.append(data[2])
            arimaResults.append(data[3])
            etsResults.append(data[4])
            realResults.append(data[6][:-1])
        lineCount += 1

averageResults = []
for i in range(len(naiveResults)):
    averageResults.append((float(naiveResults[i])+float(arResults[i])+float(armaResults[i])
                           +float(arimaResults[i])+float(etsResults[i]))/5)

errorNaive = []
errorAR = []
errorARMA = []
errorARIMA = []
errorETS = []
for i in range(len(naiveResults)):
    errorNaive.append(float(realResults[i])-float(naiveResults[i]))
    errorAR.append(float(realResults[i]) - float(arResults[i]))
    errorARMA.append(float(realResults[i]) - float(armaResults[i]))
    errorARIMA.append(float(realResults[i]) - float(arimaResults[i]))
    errorETS.append(float(realResults[i]) - float(etsResults[i]))

errorResults = []
for i in range(len(averageResults)):
    errorResults.append(float(realResults[i])-float(averageResults[i]))

with open(writePath,'w') as f:
    for i in range(1, len(averageResults)):
        line = str(errorNaive[i-1]) + ',' + str(errorAR[i-1]) + ',' + str(errorARMA[i-1]) + ',' + str(errorARIMA[i-1]) + ',' + str(errorETS[i-1]) + ',' + \
               str(errorResults[i])
        f.write(line+'\n')


