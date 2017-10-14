#For quick testing...

import csv

list = [0,1,2,3]

with open('test.csv','wb') as f:
    writer = csv.writer(f)
    for ele in list:
        writer.writerows([["abcd",ele]])