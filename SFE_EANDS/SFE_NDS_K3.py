import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import seed
from numpy.random import randint
from sklearn.model_selection import cross_val_score
import scipy.io as sio
# load data


UR = 0.3
UR_Max = 0.3
UR_Min = 0.001
Max_FEs = 6000
Max_Run = 30


def fit(xtrain, ytrain, kk):
    k = 5
    if len(kk) == 0:
        seed(1)
        kk = randint(1, np.size(xtrain, 1))
    sf = []
    for i in range(0, np.size(xtrain, 1)):
        if kk[i] == 1:
            sf.append(i)
    pos = np.transpose(sf)
    model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    X = xtrain[:, pos]
    scores = cross_val_score(model, X, ytrain, cv=5)
    cost = sum(scores) / k
    return cost * 100


def Distance(X1, X2):
    dis = 0
    dim = len(X1)
    for i in range(dim):
        dis += abs(X1[i] - X2[i])
    return dis / dim


def SFE_run(Input, Target):
    global UR, Max_FEs, Max_Run
    Cost = np.zeros([Max_FEs, Max_Run])
    Subscale = np.zeros([Max_FEs, Max_Run])
    Run = 1
    Nvar = np.size(Input, 1)

    while Run <= Max_Run:
        X = np.random.randint(0, 2, Nvar)
        Fit_X = fit(Input, Target, X)

        Best_X = np.copy(X)
        Best_Fit_X = Fit_X
        FEs = 1
        while FEs <= Max_FEs:
            X_New = np.copy(X)
            U_Index = np.where(X == 1)
            NUSF_X = np.size(U_Index, 1)

            UN = math.ceil(UR*Nvar)
            K1 = np.random.randint(0, NUSF_X, UN)
            res = np.array([*set(K1)])
            res1 = np.array(res)
            K = U_Index[0][[res1]]
            X_New[K] = 0

            if np.sum(X_New) == 0:
                S_Index = np.where(X == 0)
                NSF_X = np.size(S_Index, 1)
                SN = 1
                K1 = np.random.randint(0, NSF_X, SN)
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                X_New = np.copy(X)
                X_New[K] = 1

            Fit_X_New = fit(Input, Target, X_New)

            if Fit_X_New >= Fit_X:
                X = np.copy(X_New)
                Fit_X = Fit_X_New
                if Fit_X_New >= Best_Fit_X:
                    Best_X = np.copy(X_New)
                    Best_Fit_X = Fit_X_New
            else:  # Distance based selection
                delta = abs((Fit_X_New - Fit_X) / 100)  # Normalized
                dis = Distance(X_New, X)  # Normalized
                threshold = np.exp(- (delta / dis))
                if np.random.rand() < threshold:
                    X = np.copy(X_New)
                    Fit_X = Fit_X_New
            UR = (UR_Max-UR_Min) * ((Max_FEs - FEs) / Max_FEs) + UR_Min  # Eq(2)
            Cost[FEs-1, Run-1] = Best_Fit_X
            Subscale[FEs-1, Run-1] = np.sum(Best_X)
            # print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format(FEs, Fit_X, np.sum(X), Run))
            FEs = FEs + 1
        Run = Run+1
    return Cost, Subscale


if __name__ == '__main__':
    if os.path.exists('./SFE_K3_Data/Acc') == False:
        os.makedirs('./SFE_K3_Data/Acc')
    if os.path.exists('./SFE_K3_Data/Scale') == False:
        os.makedirs('./SFE_K3_Data/Scale')
    """
    Optimization with .csv file
    """
    Datasets = ['colon', 'CML treatment', 'leukemia', 'ALL_AML_4']
    for dataset in Datasets:
        data = pd.read_csv('../Datasets/CSV/' + dataset + '.csv').values
        Input = stats.zscore(np.asarray(data[:, 0: -1]))
        Target = np.asarray(data[:, -1])

        Acc_trial, Sub_trial = SFE_run(Input, Target)
        np.savetxt('./SFE_K3_Data/Acc/{}.csv'.format(dataset), Acc_trial.T, delimiter=",")
        np.savetxt('./SFE_K3_Data/Scale/{}.csv'.format(dataset), Sub_trial.T, delimiter=",")

    """
    Optimization with .mat file
    """
    Datasets = ['ALLAML', 'CLL_SUB_111', 'GLIOMA', 'GLI_85', 'Leukemia_1',
                'Leukemia_3', 'lung', 'lymphoma', 'nci9', 'ORL', 'orlraws10P',
                'Prostate_GE', 'SCBCT', 'SMK_CAN_187', 'TOX_171', 'warpAR10P', 'Yale']
    for dataset in Datasets:
        data = sio.loadmat('../Datasets/Mat/' + dataset + '.mat')['X']
        label = sio.loadmat('../Datasets/Mat/' + dataset + '.mat')['Y'][:, 0]

        Input = stats.zscore(np.asarray(data))
        Target = np.asarray(label)

        Acc_trial, Sub_trial = SFE_run(Input, Target)
        np.savetxt('./SFE_K3_Data/Acc/{}.csv'.format(dataset), Acc_trial.T, delimiter=",")
        np.savetxt('./SFE_K3_Data/Scale/{}.csv'.format(dataset), Sub_trial.T, delimiter=",")




