from os import listdir
import matplotlib.pyplot as plt

dataPath = '/home/minh/PycharmProjects/Ensemble/preprocessed/EDGAR_11-12/'

fileList = listdir(dataPath)
fileList.sort()
data = []
xAxis = []

for filename in fileList:
    with open(dataPath + filename, 'r') as f:
        for line in f:
            data.append(float(line.split(',')[1]))

for i in range(len(data)):
    xAxis.append(i)

plt.plot(xAxis,data)
plt.show()