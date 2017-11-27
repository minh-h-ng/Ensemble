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
import numpy as np
import copy

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

    elif (dataset=='Kyoto'):
        preprocessedDir = '/home/minh/PycharmProjects/Ensemble/preprocessed/Kyoto'
        processedFile = '/home/minh/PycharmProjects/Ensemble/processed/Kyoto/access.csv'
        fileList = listdir(preprocessedDir)
        fileList.sort()
        dataList = []
        for fileName in fileList:
            with open(preprocessedDir + "/" + fileName, 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    dataList.append(line)
        with open(processedFile, 'w') as f:
            writer = csv.writer(f)
            for line in dataList:
                writer.writerows([line])

    elif (dataset=='CRAN'):
        preprocessedDir = '/home/minh/PycharmProjects/Ensemble/preprocessed/CRAN'
        processedFile = '/home/minh/PycharmProjects/Ensemble/processed/CRAN/access.csv'
        fileList = listdir(preprocessedDir)
        fileList.sort()
        dataList = []
        for fileName in fileList:
            with open(preprocessedDir + "/" + fileName, 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    dataList.append(line)
        with open(processedFile, 'w') as f:
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
        preprocessedFolder = '/home/minh/PycharmProjects/Ensemble/preprocessed/EDGAR'
        dataFolder = '/home/minh/Desktop/edgar'
        chdir(dataFolder)

        for fileName in listdir('.'):
            chdir(dataFolder)
            curCount = 0
            requestList = []
            totalCount = 0
            with open(fileName,'r') as f:
                reader = csv.reader(f)
                count = 0
                for line in reader:
                    totalCount+=1
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
                    if count%1000000==0:
                        print("Number of lines processed:",count)
            requestList.append(curCount+1)

            print('totalCount:',totalCount)
            print('total requestList:',np.sum(requestList))

            print('length requestList:',len(requestList))
            print('lastTimeStamp:',lastTimeStamp)

            hourlyRequest = []
            for i in range(24):
                hourlyRequest.append(0)

            for i in range(len(requestList)):
                curHour = int(i/3600)
                if requestList[i]>hourlyRequest[curHour]:
                    hourlyRequest[curHour]=requestList[i]

            chdir(preprocessedFolder)
            curTime = startTime
            timeDelta = timedelta(hours=1)
            with open(fileName,'w') as f:
                writer = csv.writer(f)
                for value in hourlyRequest:
                    writer.writerows([[curTime.strftime("%Y-%m-%d %H:%M:%S"),value]])
                    curTime += timeDelta
    elif (dataset=='MACCDC2012'):
        dataFile = '/home/minh/Desktop/maccdc2012/http.log'
        preprocessedFile = '/home/minh/PycharmProjects/Ensemble/preprocessed/MACCDC2012/log.csv'
        # "Time Reference": midnight UTC of January 1, 1970
        referenceTime = datetime.strptime("1970-01-01 00:00:00","%Y-%m-%d %H:%M:%S")
        firstTime = None
        prevTime = None
        timeCount = 0
        requests = []
        requestList = []
        count = 0
        lastTime = None
        with open(dataFile,'r') as f:
            for line in f:
                count+=1
                lineParts = line.split('.')
                if prevTime==None:
                    firstTime = referenceTime + timedelta(seconds=int(lineParts[0]))
                    prevTime = referenceTime + timedelta(seconds=int(lineParts[0]))
                    timeCount+=1
                else:
                    curTime = referenceTime + timedelta(seconds=int(lineParts[0]))
                    timeDiff = int((curTime-prevTime).total_seconds())
                    if prevTime > curTime:
                        diff = int((curTime-firstTime).total_seconds())
                        if int(diff/3600)<len(requests):
                            requests[int(diff/3600)][diff%3600]+=1
                        else:
                            requestList[diff%3600]+=1
                    elif timeDiff==0:
                        timeCount+=1
                    else:
                        requestList.append(timeCount)
                        if len(requestList)==3600:
                            requests.append(copy.deepcopy(requestList))
                            requestList = []
                        timeCount = 1
                        for i in range(timeDiff-1):
                            requestList.append(0)
                            if len(requestList)==3600:
                                requests.append(copy.deepcopy(requestList))
                                requestList = []
                        prevTime = curTime
                if lastTime==None or referenceTime + timedelta(seconds=int(lineParts[0]))>lastTime:
                    lastTime = referenceTime + timedelta(seconds=int(lineParts[0]))
                if count%100000==0:
                    print('Number of lines processed:',count)
        requestList.append(timeCount)
        requests.append(copy.deepcopy(requestList))

        print('firstTime:', firstTime)
        print('lastTime:', lastTime)
        print('count:', count)
        totalSum = 0
        totalLength = 0
        for i in range(len(requests)):
            totalSum+=np.sum(requests[i])
            totalLength+=len(requests[i])
        print('total sum:',totalSum)
        print('total length:',totalLength)
        print('time diff:',(lastTime-firstTime).total_seconds())

        """firstTime: 2012-03-16 12:30:00
        lastTime: 2012-03-17 20:46:54
        count: 2048442
        total sum: 2048443
        total length: 116215
        time diff: 116214.0"""

        """for i in range(len(requests)):
            print('length list:',len(requests[i]))"""

        hourlyRequest = []
        for i in range(len(requests)):
            curList = requests[i]
            hourlyRequest.append(np.max(curList))

        """for i in range(24):
            hourlyRequest.append(0)

        for i in range(len(requestList)):
            curHour = int(i / 3600)
            if requestList[i] > hourlyRequest[curHour]:
                hourlyRequest[curHour] = requestList[i]"""

        curTime = firstTime
        with open(preprocessedFile,'w') as f:
            writer = csv.writer(f)
            for value in hourlyRequest:
                writer.writerows([[curTime.strftime("%Y-%m-%d %H:%M:%S"), value]])
                curTime += timedelta(hours=1)


        """dataFile = '/home/minh/Desktop/maccdc2012/http.log'
        count = 0
        with open(dataFile,'r') as f:
            for line in f:
                if count<=1:
                    print('line:',line)
                else:
                    break
                count+=1
        startTime = datetime.strptime("1970-01-01 00:00:00","%Y-%m-%d %H:%M:%S")
        timeDiff = timedelta(seconds=1331901000)
        curTime = startTime + timeDiff
        print('startTime:',startTime)
        print('curTime:',curTime)"""
    elif (dataset=='Kyoto'):
        dataFolder = '/home/minh/Desktop/kyoto2015/'
        preprocessedFolder = '/home/minh/PycharmProjects/Ensemble/preprocessed/Kyoto/'
        fileList = listdir(dataFolder)
        fileList.sort()
        print('fileList:',fileList)

        requestList = []
        for i in range(86400):
            requestList.append(0)

        count=0
        for fileName in fileList:
            curDay = fileName[:4] + '-' + fileName[4:6] + '-' + fileName[6:8]
            filePath = dataFolder + fileName
            with open(filePath,'r') as f:
                for line in f:
                    count+=1
                    lineParts = line.split()
                    h,m,s = lineParts[-2].split(':')
                    noOfSeconds = int(h) * 3600 + int(m) * 60 + int(s)
                    requestList[noOfSeconds] += 1
                    if count%100000==0:
                        print('number of lines processed:',count)

            hourlyRequest = []
            for i in range(24):
                hourlyRequest.append(0)

            for i in range(len(requestList)):
                curHour = int(i / 3600)
                if requestList[i] > hourlyRequest[curHour]:
                    hourlyRequest[curHour] = requestList[i]

            startTime = datetime.strptime(curDay + " " + "00:00:00", "%Y-%m-%d %H:%M:%S")
            timeDelta = timedelta(hours=1)
            curTime = startTime

            with open(preprocessedFolder + fileName[:8]+'.csv','w') as f:
                writer = csv.writer(f)
                for value in hourlyRequest:
                    writer.writerows([[curTime.strftime("%Y-%m-%d %H:%M:%S"), value]])
                    curTime += timeDelta
            print('file processed:', fileName)
    elif (dataset=='CRAN'):
        dataFolder = '/home/minh/Desktop/CRAN/Extracted/'
        preprocessedFolder = '/home/minh/PycharmProjects/Ensemble/preprocessed/CRAN/'
        fileList = listdir(dataFolder)
        fileList.sort()
        print('fileList:', fileList)

        requestList = []
        for i in range(86400):
            requestList.append(0)

        for fileName in fileList:
            count = 0
            curDay = fileName[:10]
            filePath = dataFolder + fileName
            with open(filePath, 'r') as f:
                for line in f:
                    count += 1
                    if count>1:
                        lineParts = line.split(',')
                        h, m, s = lineParts[1][1:-1].split(':')
                        noOfSeconds = int(h) * 3600 + int(m) * 60 + int(s)
                        requestList[noOfSeconds] += 1
                        if count % 100000 == 0:
                            print('number of lines processed:', count)

            hourlyRequest = []
            for i in range(24):
                hourlyRequest.append(0)

            for i in range(len(requestList)):
                curHour = int(i / 3600)
                if requestList[i] > hourlyRequest[curHour]:
                    hourlyRequest[curHour] = requestList[i]

            startTime = datetime.strptime(curDay + " " + "00:00:00", "%Y-%m-%d %H:%M:%S")
            timeDelta = timedelta(hours=1)
            curTime = startTime

            with open(preprocessedFolder + fileName[:4] + fileName[5:7] + fileName[8:10] + '.csv', 'w') as f:
                writer = csv.writer(f)
                for value in hourlyRequest:
                    writer.writerows([[curTime.strftime("%Y-%m-%d %H:%M:%S"), value]])
                    curTime += timeDelta
            print('file processed:', fileName)

#preprocess('CRAN')
#process('CRAN')