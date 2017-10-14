"""
All necessary utilities for the project
"""

from os import chdir, getcwd
import json
import csv
import numpy as np

with open('config.json','r') as f:
    config = json.load(f)

def gotoTopDir():
    topDir = config['PROJECT']['TOP_DIR'].encode('utf-8')
    curDirList = getcwd().split('/')
    for i in xrange(len(curDirList)):
        if curDirList[i]==topDir:
            for j in xrange(len(curDirList)-1-i):
                chdir('..')
            return
    print('Cannot find top directory!')