"""
All types of necessary preprocessing work: convert into data per hour, make graph, etc...
"""

from os import chdir, listdir
from datetime import datetime
from datetime import timedelta
import os
import csv
import Utilities
import json
import matplotlib.pyplot as plt

with open('config.json','r') as f:
    config = json.load(f)

def process(dataset):
    if (dataset=='EDGAR'):
        Utilities.gotoTopDir()
        preprocessedDir = config['PREPROCESSED']['EDGAR_DIR']
        processedLocation = config['PROCESSED']['EDGAR_LOCATION']
        fileList = listdir(preprocessedDir)
        fileList.sort()
        dataList = []
        for fileName in fileList:
            with open(preprocessedDir + "/" + fileName, 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    dataList.append(line)
        with open(processedLocation,'w') as f:
            writer = csv.writer(f)
            for line in dataList:
                writer.writerows([line])

def graph(dataset):
    if (dataset=='EDGAR'):
        processedLocation = config['PROCESSED']['EDGAR_LOCATION'].encode('UTF-8')
        dataList = []
        with open(processedLocation,'rb') as f:
            reader = csv.reader(f)
            for line in reader:
                dataList.append(float(line[1]))
        xAxis = []
        for i in range(len(dataList)):
            xAxis.append(i)
        plt.plot(xAxis,dataList)
        plt.show()


#old code, need to update paramters to config.json if used later
def preprocess(dataset):
    if (dataset=='EDGAR'):
        Utilities.gotoTopDir()
        chdir('Backup/EDGAR')

        for fileName in listdir('.'):
            Utilities.gotoTopDir()
            chdir('Backup/EDGAR')
            curCount = 0
            requestList = []

            with open(fileName,'r') as f:
                reader = csv.reader(f)
                count = 0
                for line in reader:
                    count+=1
                    if count>1:
                        if count==2:
                            startTime = datetime.strptime(line[1]+" " + line[2],"%Y-%m-%d %H:%M:%S")
                            curTime = datetime.strptime(line[1]+" " + line[2],"%Y-%m-%d %H:%M:%S")
                            dateInfo = line[1]
                        elif line[1]!=dateInfo:
                            print('problem with date in line:',line)
                        time = datetime.strptime(line[1]+" " + line[2],"%Y-%m-%d %H:%M:%S")
                        timeDelta = time-curTime
                        if timeDelta.seconds==0:
                            curCount += 1
                        else:
                            requestList.append(curCount+1)
                            for i in range(timeDelta.seconds-1):
                                requestList.append(0)
                            curTime = time
                            curCount = 0
                        lastTimeStamp = time
                    if count%100000==0:
                        print("Number of lines processed:",count)
            requestList.append(curCount)

            print('length requestList:',len(requestList))
            print('lastTimeStamp:',lastTimeStamp)

            hourlyRequest = []
            for i in range(24):
                hourlyRequest.append(0)

            for i in range(len(requestList)):
                curHour = int(i/3600)
                if requestList[i]>hourlyRequest[curHour]:
                    hourlyRequest[curHour]=requestList[i]

            Utilities.gotoTopDir()
            chdir('preprocessed/EDGAR')
            curTime = startTime
            timeDelta = timedelta(hours=1)
            with open(fileName,'w') as f:
                writer = csv.writer(f)
                for value in hourlyRequest:
                    writer.writerows([[curTime.strftime("%Y-%m-%d %H:%M:%S"),value]])
                    curTime += timeDelta

#preprocess('EDGAR')
#process('EDGAR')
#graph('EDGAR')