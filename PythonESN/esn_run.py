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
    # Run in parallel and store result in a numpy array
    X, Y = esnet.load_from_text(args.data)
    predictions = []
    count = 0
    for i in range(150,len(X)):
        Xtr, Ytr, _, _, Xte, Yte, Yscaler = esnet.generate_datasets(X[:i], Y[:i])
        Yhat, error = esnet.run_from_config(Xtr, Ytr, Xte, Yte, config)
        Yhat = np.rint(Yscaler.inverse_transform(Yhat))
        #print('predictions:',Yhat)
        #print('error:',error)
        predictions.append(Yhat[len(Yhat)-1][0])
        count+=1
        if count%100==0:
            print('predictions made:',count)

    curPath = os.getcwd().split('/')
    writePath = ''
    for i in range(len(curPath)):
        writePath += curPath[i] + '/'
    writePath += 'predictions/predictions'

    with open(writePath,'w') as f:
        for value in predictions:
            f.write(str(value) + '\n')

if __name__ == "__main__":
    main()
