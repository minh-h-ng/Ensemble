"""
Download datasets
"""

import urllib
import Utilities
from os import chdir
import json
from bs4 import BeautifulSoup
import requests

with open('config.json','r') as f:
    config = json.load(f)

def downloadALM():
    sourceLocation = config['SOURCE']['ALM_LOCATION'].encode('UTF-8')
    sourceURL = config['SOURCE']['ALM_URL'].encode('UTF-8')
    Utilities.gotoTopDir()
    sourcefile = urllib.URLopener()
    sourcefile.retrieve(sourceURL, sourceLocation)

def downloadSVDS():
    sourceLocation = config['SOURCE']['SVDS_LOCATION'].encode('UTF-8')
    sourceURL = config['SOURCE']['SVDS_URL'].encode('UTF-8')
    Utilities.gotoTopDir()
    sourcefile = urllib.URLopener()
    sourcefile.retrieve(sourceURL,sourceLocation)

def downloadSecRepo():
    sourceDir = config['SOURCE']['SECREPO_DIR'].encode('UTF-8')
    sourceURL = config['SOURCE']['SECREPO_URL'].encode('UTF-8')
    Utilities.gotoTopDir()
    soup = BeautifulSoup(requests.get(sourceURL).text)
    hrefs = []
    for a in soup.find_all('a'):
        hrefs.append(a['href'])
    for href in hrefs:
        fileType = href[:6]
        if fileType=='access':
            link = sourceURL + href
            sourcefile = urllib.URLopener()
            sourcefile.retrieve(link,sourceDir + href)
