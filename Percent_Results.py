import csv
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('top_dir', type=str, help='Path to project directory')
args = parser.parse_args()

finalCRAN = args.top_dir + '/final_results/cran_10_12.csv'
finalEDGAR = args.top_dir + '/final_results/edgar_10_12.csv'
finalKyoto = args.top_dir + '/final_results/kyoto_10_12.csv'

outCRAN = args.top_dir + '/final_results/cran_percent_10_12.csv'
outEDGAR = args.top_dir + '/final_results/edgar_percent_10_12.csv'
outKyoto = args.top_dir + '/final_results/kyoto_percent_10_12.csv'

outList = [outCRAN,outEDGAR,outKyoto]
finalList = [finalCRAN,finalEDGAR,finalKyoto]

def percent(predictList,realList):
    results = []
    for i in range(len(realList)):
        results.append((float(predictList[i])-float(realList[i]))/float(realList[i])*100)
    return results

def doReals(dataFile):
    reals = []
    with open(dataFile, 'r') as f:
        count = 0
        reader = csv.reader(f)
        for line in reader:
            count += 1
            if count == 1:
                names = line
            if count > 1:
                reals.append(line[0])
    names = names[1:]
    return names,reals

def doComponents(dataFile):
    components = []
    with open(dataFile, 'r') as f:
        count = 0
        reader = csv.reader(f)
        for line in reader:
            count+=1
            if count==1:
                compLength = len(line)
                for i in range(compLength-1):
                    aList = []
                    components.append(aList)
            if count>1:
                for j in range(1,compLength):
                    components[j-1].append(line[j])
    return components

def doPercent(components,reals):
    compPercent = []
    for i in range(len(components)):
        compPercent.append(percent(components[i],reals))
    return compPercent

def main():
    for i in range(len(finalList)):
        names,reals = doReals(finalList[i])
        components = doComponents(finalList[i])
        compPercent = doPercent(components,reals)
        with open(outList[i],'w') as f:
            outName = ''
            for j in range(len(names)):
                if j==len(names)-1:
                    outName+=names[j]
                else:
                    outName+=names[j]+','
            f.write(outName)
            f.write('\n')
            for j in range(len(compPercent[0])):
                outData = ''
                for k in range(len(compPercent)):
                    if k==len(compPercent)-1:
                        outData+=str(compPercent[k][j])
                    else:
                        outData+=str(compPercent[k][j])+','
                f.write(outData)
                f.write('\n')

        print('names:',names)
        print('reals:',len(reals))
        print('components:',len(components),len(components[0]))
        print('components:', len(compPercent), len(compPercent[0]))
        print(compPercent)

if __name__ == "__main__":
    main()