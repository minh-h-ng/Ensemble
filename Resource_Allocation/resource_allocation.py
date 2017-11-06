import math

class ResourceAllocation:
    def __init__(self, responseTime, processingRate):
        """
        Initialization for resource allocation
        :param responseRate: response (SLA)
        :param processingRate: processing capacity of servers
        """
        self.responseTime = responseTime
        self.processingRate = processingRate

    def resourceAllocation(self, predictions, reals):
        """
        :param predictions: list of predictions from an algorithm
        :param reals: real web request values
        :return (over-provisioned resources, under-provisioned resources)
        """
        overResources = 0
        underResources = 0
        for i in range(len(predictions)):
            usedResources = math.ceil((self.responseTime*predictions[i])/(self.responseTime*self.processingRate-1))
            necessaryResources = math.ceil((self.responseTime*reals[i])/(self.responseTime*self.processingRate-1))
            #print('necessaryResources:',necessaryResources)
            if usedResources>necessaryResources:
                overResources+=usedResources-necessaryResources
            else:
                underResources+=necessaryResources-usedResources
        return (overResources, underResources)

if __name__ == '__main__':

    resourceAllocation = ResourceAllocation(0.4, 10)
    dataPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar'
    predictionPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/predictions/predictions_edgar_historical_enet_identity'

    naivePredictions = []
    arPredictions = []
    armaPredictions = []
    arimaPredictions= []
    etsPredictions = []
    reals = []

    count = 0
    with open(dataPath, 'r') as f:
        for line in f:
            count += 1
            if count > 1:
                data = line.split(',')
                naivePredictions.append(float(data[0]))
                arPredictions.append(float(data[1]))
                armaPredictions.append(float(data[2]))
                arimaPredictions.append(float(data[3]))
                etsPredictions.append(float(data[4]))
                reals.append(float(data[6][:-1]))

    overResources, underResources = resourceAllocation.resourceAllocation(naivePredictions,reals)
    print('Naive over, under-provision, total resources:',overResources,underResources,overResources+underResources)

    overResources, underResources = resourceAllocation.resourceAllocation(arPredictions, reals)
    print('AR over, under-provision, total resources:', overResources, underResources,overResources+underResources)

    overResources, underResources = resourceAllocation.resourceAllocation(armaPredictions, reals)
    print('ARMA over, under-provision, total resources:', overResources, underResources,overResources+underResources)

    overResources, underResources = resourceAllocation.resourceAllocation(arimaPredictions, reals)
    print('ARIMA over, under-provision, total resources:', overResources, underResources,overResources+underResources)

    overResources, underResources = resourceAllocation.resourceAllocation(etsPredictions, reals)
    print('ETS over, under-provision, total resources:', overResources, underResources,overResources+underResources)

    esnPredictions = []
    with open(predictionPath, 'r') as f:
        for line in f:
            if float(line)>1:
                esnPredictions.append(float(line))
            else:
                esnPredictions.append(1)

    overResources, underResources = resourceAllocation.resourceAllocation(esnPredictions, reals)
    print('ESN over & under-provision resources:', overResources, underResources,overResources+underResources)