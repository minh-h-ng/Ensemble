import lightgbm as lgb
#import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

cranFile = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/cran_10_12'
edgarFile = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar_10_12'
kyotoFile = '/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/kyoto_10_12'

def readDataset(dataFilfe):
    naiveList = []
    arList = []
    armaList = []
    arimaList = []
    etsList = []
    realList = []
    with open(cranFile,'r') as f:
        count = 0
        for line in f:
            count+=1
            if count>1:
                lineParts = line.split(',')
                naiveList.append(float(lineParts[0]))
                arList.append(float(lineParts[1]))
                armaList.append(float(lineParts[2]))
                arimaList.append(float(lineParts[3]))
                etsList.append(float(lineParts[4]))
                realList.append(float(lineParts[-1][:-1]))
    testingSize = 240
    X = []
    y = []
    for i in range(len(arList)):
        aList = []
        aList.append(naiveList[i])
        #aList.append(arList[i])
        #aList.append(armaList[i])
        #aList.append(arimaList[i])
        #aList.append(etsList[i])
        X.append(aList)
        y.append(realList[i])
    X_train = X[:-240]
    X_test = X[-240:]
    y_train = y[:-240]
    y_test = y[-240:]

    Xscaler = StandardScaler()
    yscaler = MinMaxScaler()

    X_train = Xscaler.fit_transform(X_train)
    y_train = np.array(y_train).reshape(-1,1)
    #y_train = y_train.reshape(-1,1)
    #print('y_train:',y_train)
    y_train = yscaler.fit_transform(y_train)

    X_test = Xscaler.transform(X_test)
    y_test = np.array(y_test).reshape(-1, 1)
    y_test = yscaler.transform(y_test)

    return X_train,X_test,y_train,y_test,Xscaler,yscaler
    #return naiveList,arList,armaList,arimaList,etsList,realList

def GBR(X_train,X_test,y_train,y_test,Xscaler,yscaler):
    print('Start training...')
    # train
    """gbm = lgb.LGBMRegressor(objective='regression',
                            num_leaves=100,
                            learning_rate=0.01,
                            n_estimators=50)"""
    gbm = lgb.LGBMRegressor(objective='regression')
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            #early_stopping_rounds=20
            )

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    y_pred = yscaler.inverse_transform(np.array(y_pred).reshape(-1,1))

    # other scikit-learn modules
    """estimator = lgb.LGBMRegressor(num_leaves=50)

    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [20, 40]
    }

    gbm = GridSearchCV(estimator, param_grid)

    gbm.fit(X_train, y_train)

    print('Best parameters found by grid search are:', gbm.best_params_)"""

    #print('y_test:',y_test)
    #print(gbm.predict(X_test))
    return y_pred

def adaBoost(X_train,X_test,y_train,y_test,Xscaler,yscaler):

    # Fit regression model
    rng = np.random.RandomState(1)
    regr_1 = DecisionTreeRegressor(max_depth=40)

    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=40),
                               n_estimators=3000, random_state=rng)

    regr_1.fit(X_train, y_train)
    regr_2.fit(X_train, y_train)

    # Predict
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    y_1 = yscaler.inverse_transform(np.array(y_1).reshape(-1,1))
    y_2 = yscaler.inverse_transform(np.array(y_2).reshape(-1,1))

    return y_2

def main():
    warnings.filterwarnings('ignore')
    X_train,X_test,y_train,y_test,Xscaler,yscaler = readDataset(cranFile)
    predictions = adaBoost(X_train,X_test,y_train,y_test,Xscaler,yscaler)
    with open('/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/cran_10_12/Ensemble_GBM_cran_10_12_1','w') as f:
        f.write('Observation,Prediction\n')
        for i in range(len(predictions)):
            f.write(str(y_test[i][0]) + ',' + str(predictions[i][0]))
            f.write('\n')

    X_train, X_test, y_train, y_test,Xscaler,yscaler = readDataset(edgarFile)
    predictions = adaBoost(X_train, X_test, y_train, y_test,Xscaler,yscaler)
    with open('/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/edgar_10_12/Ensemble_GBM_edgar_10_12_1','w') as f:
        f.write('Observation,Prediction\n')
        for i in range(len(predictions)):
            f.write(str(y_test[i][0]) + ',' + str(predictions[i][0]))
            f.write('\n')

    X_train, X_test, y_train, y_test,Xscaler,yscaler = readDataset(kyotoFile)
    predictions = adaBoost(X_train, X_test, y_train, y_test,Xscaler,yscaler)
    with open('/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/kyoto_10_12/Ensemble_GBM_kyoto_10_12_1','w') as f:
        f.write('Observation,Prediction\n')
        for i in range(len(predictions)):
            f.write(str(y_test[i][0]) + ',' + str(predictions[i][0]))
            f.write('\n')

if __name__ == "__main__":
    main()