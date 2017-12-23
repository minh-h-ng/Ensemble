import csv

accuracyCRAN = '/home/minh/PycharmProjects/Ensemble/final_results/cran_accuracy_10_12.csv'
accurayEDGAR = '/home/minh/PycharmProjects/Ensemble/final_results/edgar_accuracy_10_12.csv'
accuracyKyoto = '/home/minh/PycharmProjects/Ensemble/final_results/kyoto_accuracy_10_12.csv'

outFile = '/home/minh/PycharmProjects/Ensemble/final_results/cost_projection.csv'

accuracyList = [accuracyCRAN,accurayEDGAR,accuracyKyoto]

testingSize = 240

def main():
    with open(accuracyList[0],'r') as f:
        reader = csv.reader(f)
        count = 0
        for line in reader:
            count+=1
            if count==1:
                names = line
            else:
                break
    cranList = [0]*len(names)
    cranList[0] = 'cran'
    edgarList = [0]*len(names)
    edgarList[0] = 'edgar'
    kyotoList = [0]*len(names)
    kyotoList[0] = 'kyoto'
    for i in range(len(accuracyList)):
        with open(accuracyList[i],'r') as f:
            reader = csv.reader(f)
            count = 0
            for line in reader:
                count+=1
                if count==1:
                    names=line
                elif line[0]=='total - 1st scenario':
                    scenario1=line
                elif line[0]=='total - 2nd scenario':
                    scenario2=line
            if i==0:
                for j in range(1,len(scenario1)):
                    cranList[j] = (float(scenario1[j]) + float(scenario2[j]))/testingSize*365
            elif i==1:
                for j in range(1,len(scenario1)):
                    edgarList[j] = (float(scenario1[j]) + float(scenario2[j]))/testingSize*365
            elif i==2:
                for j in range(1,len(scenario1)):
                    kyotoList[j] = (float(scenario1[j]) + float(scenario2[j]))/testingSize*365
    totalList = []
    totalList.append('total')
    for i in range(1,len(cranList)):
        totalList.append((cranList[i]+edgarList[i]+kyotoList[i])/3)
    with open(outFile,'w') as f:
        writer=csv.writer(f)
        writer.writerows([names])
        writer.writerows([cranList])
        writer.writerows([edgarList])
        writer.writerows([kyotoList])
        writer.writerows([totalList])

if __name__ == "__main__":
    main()