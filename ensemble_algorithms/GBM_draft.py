import lightgbm as lgb
#import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings

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
        aList.append(arList[i])
        aList.append(armaList[i])
        aList.append(arimaList[i])
        aList.append(etsList[i])
        X.append(aList)
        y.append(realList[i])
    X_train = X[:-240]
    X_test = X[-240:]
    y_train = y[:-240]
    y_test = y[-240:]
    return X_train,X_test,y_train,y_test
    #return naiveList,arList,armaList,arimaList,etsList,realList

def GBR(X_train,X_test,y_train,y_test):
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
            early_stopping_rounds=20)

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    # feature importances
    print('Feature importances:', list(gbm.feature_importances_))

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
    return gbm.predict(X_test)

def main():
    warnings.filterwarnings('ignore')
    X_train,X_test,y_train,y_test = readDataset(cranFile)
    predictions = GBR(X_train,X_test,y_train,y_test)
    with open('/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/cran_10_12/Ensemble_GBM_cran_10_12_1','w') as f:
        f.write('Observation,Prediction\n')
        for i in range(len(predictions)):
            f.write(str(y_test[i]) + ',' + str(predictions[i]))
            f.write('\n')

    X_train, X_test, y_train, y_test = readDataset(edgarFile)
    predictions = GBR(X_train, X_test, y_train, y_test)
    with open('/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/edgar_10_12/Ensemble_GBM_edgar_10_12_1','w') as f:
        f.write('Observation,Prediction\n')
        for i in range(len(predictions)):
            f.write(str(y_test[i]) + ',' + str(predictions[i]))
            f.write('\n')

    X_train, X_test, y_train, y_test = readDataset(kyotoFile)
    predictions = GBR(X_train, X_test, y_train, y_test)
    with open('/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/results/kyoto_10_12/Ensemble_GBM_kyoto_10_12_1','w') as f:
        f.write('Observation,Prediction\n')
        for i in range(len(predictions)):
            f.write(str(y_test[i]) + ',' + str(predictions[i]))
            f.write('\n')

if __name__ == "__main__":
    main()