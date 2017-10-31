import argparse
import json
import logging
import numpy as np
import os

import esnet

# Initialize logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

###############################################################################################
# The next part needs to be in the global scope, since all workers
# need access to these variables. I got pickling problems when using
# them as arguments in the evaluation function. I couldn't pickle the
# partial function for some reason, even though it should be supported.
############################################################################
# Parse input arguments
############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("data", help="path to data file", type=str)
parser.add_argument("esnconfig", help="path to ESN config file", type=str)
parser.add_argument("nexp", help="number of runs", type=int)
args = parser.parse_args()

############################################################################
# Read config file
############################################################################
config = json.load(open(args.esnconfig + '.json', 'r'))

############################################################################
# Load data
############################################################################
# If the data is stored in a directory, load the data from there. Otherwise,
# load from the single file and split it.
"""if os.path.isdir(args.data):
    Xtr, Ytr, _, _, Xte, Yte = esnet.load_from_dir(args.data)

else:
    X, Y = esnet.load_from_text(args.data)

    # Construct training/test sets
    Xtr, Ytr, _, _, Xte, Yte, Yscaler = esnet.generate_datasets(X, Y)"""

def main():
    averages = []
    predictions_error = []
    predictions = []
    reals = []
    startPoint = 4

    #For the first few predictions, use the last error as prediction
    dataPath = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar'
    count = 0
    with open(dataPath,'r') as f:
        for line in f:
            if count>0:
                data = line.split(',')
                total = 0
                for i in range(5):
                    total += float(data[i])
                averages.append(total/5)
                reals.append(float(data[6]))
            count+=1
    """count = 0
    with open(dataPath,'r') as f:
        for line in f:
            if count>0:
                if count>(startPoint+2):
                    break
                else:
                    data = line.split(',')
                    total = 0
                    for i in range(5):
                        total += float(data[i])
                    predictions.append(float(data[6])-total/5)
            count+=1
    print('predictions:',predictions)"""

    # Run in parallel and store result in a numpy array
    X, Y = esnet.load_from_text(args.data)
    count = 0
    for i in range(startPoint,len(X)):
        Xtr, Ytr, _, _, Xte, Yte, Yscaler = esnet.generate_datasets(X[:i], Y[:i])
        if (i-2)<100:
            config['n_drop'] = i-2
        Yhat, error = esnet.run_from_config(Xtr, Ytr, Xte, Yte, config)
        Yhat = np.rint(Yscaler.inverse_transform(Yhat))
        #print('predictions:',Yhat)
        #print('error:',error)
        predictions_error.append(Yhat[len(Yhat)-1][0])
        count+=1
        if count%100==0:
            print('predictions made:',count)

    curPath = os.getcwd().split('/')
    writePath = ''
    for i in range(len(curPath)):
        writePath += curPath[i] + '/'
    configs = args.esnconfig.split('/')
    writePath += 'predictions/predictions_' + configs[-1]

    for i in range(startPoint+2):
        predictions.append(averages[i])

    for i in range(len(predictions_error)):
        predictions.append(predictions_error[i]+averages[len(averages)-len(predictions_error)+i])

    for i in range(len(predictions)):
        if predictions[i]<1:
            predictions[i]=1
        else:
            predictions[i]=np.rint(predictions[i])

    with open(writePath,'w') as f:
        for value in predictions:
            f.write(str(value) + '\n')

if __name__ == "__main__":
    main()
