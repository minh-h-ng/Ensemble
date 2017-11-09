
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
    for i in range(5, len(averageResults)):
        """line = str(naiveResults[i-5]) + ',' + str(arResults[i-5]) + ',' + str(armaResults[i-5]) + ',' + str(arimaResults[i-5]) + ',' + str(etsResults[i-5]) + ',' + \
               str(realResults[i-5]) + ',' + \
               str(naiveResults[i-4]) + ',' + str(arResults[i-4]) + ',' + str(armaResults[i-4]) + ',' + str(arimaResults[i-4]) + ',' + str(etsResults[i-4]) + ',' + \
               str(realResults[i-4]) + ',' + \
               str(naiveResults[i-3]) + ',' + str(arResults[i-3]) + ',' + str(armaResults[i-3]) + ',' + str(arimaResults[i-3]) + ',' + str(etsResults[i-3]) + ',' + \
               str(realResults[i-3]) + ',' + \
               str(naiveResults[i-2]) + ',' + str(arResults[i-2]) + ',' + str(armaResults[i-2]) + ',' + str(arimaResults[i-2]) + ',' + str(etsResults[i-2]) + ',' + \
               str(realResults[i-2]) + ',' + \
               str(naiveResults[i-1]) + ',' + str(arResults[i-1]) + ',' + str(armaResults[i-1]) + ',' + str(arimaResults[i-1]) + ',' + str(etsResults[i-1]) + ',' + \
               str(realResults[i-1]) + ',' + \
               str(naiveResults[i]) + ',' + str(arResults[i]) + ',' + str(armaResults[i]) + ',' + str(arimaResults[i]) + ',' + str(etsResults[i]) + ',' + \
               str(realResults[i])
        f.write(line+'\n')"""
    """for i in range(2, len(averageResults)):
        line = str(naiveResults[i - 2]) + ',' + str(arResults[i - 2]) + ',' + str(armaResults[i - 2]) + ',' + \
               str(arimaResults[i - 2]) + ',' + str(etsResults[i - 2]) + ',' + str(realResults[i - 2]) + ',' + \
               str(naiveResults[i - 1]) + ',' + str(arResults[i - 1]) + ',' + str(armaResults[i - 1]) + ',' + \
               str(arimaResults[i - 1]) + ',' + str(etsResults[i - 1]) + ',' + str(realResults[i - 1]) + ',' + \
               str(realResults[i])
        f.write(line+'\n')"""
    for i in range(5, len(averageResults)):
        line = str(realResults[i-5]) + ',' + str(realResults[i-4]) + ',' + str(realResults[i-3]) + ',' + str(realResults[i-2]) + ',' + str(realResults[i-1]) + ',' + str(realResults[i])
        f.write(line + '\n')


