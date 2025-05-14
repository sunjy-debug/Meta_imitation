# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import numpy as np
import scipy.io as sio
import pickle as pkl
from datatools import *




def main(dataset,dimension,flipRate,expID,z_11,z_22):

    print ("=======================================================================================")
    print ("Dataset:{},expID:{},flipRate:{} start!!!".format(dataset,expID,flipRate))
    print ("=======================================================================================")

    data,parts = readData(dataset)
    trainIndex,valIndex,testIndex = dataSplit(parts)
    trainX, valX, testX = covariateTransform(data,dimension,trainIndex,valIndex,testIndex)
    trainA, valA, testA = adjMatrixSplit(data,trainIndex,valIndex,testIndex,dataset)



    betaConfounding = 1 # effect of features X to T (confounding1)
    betaNeighborConfounding = 1# effect of Neighbor features to T (confounding2)
    betaTreat2Outcome = 1 # effect of treatment to potential outcome
    betaCovariate2Outcome = 1 #effect of features to potential outcome (confounding1)
    betaNeighborCovariate2Outcome = 0.5 #effect of Neighbor features to potential outcome
    betaNeighborTreatment2Outcome = 1 #effect of interence
    betaNoise = 0.1 #noise

    w_z1 = 2 * np.random.random_sample((dimension)) - 1 #effect of X to T 

    T_train,meanT_train = treatmentSimulation(w_z1,trainX,trainA,betaConfounding,betaNeighborConfounding)
    T_val,meanT_val = treatmentSimulation(w_z1,valX,valA,betaConfounding,betaNeighborConfounding)
    T_test,meanT_test = treatmentSimulation(w_z1,testX,testA,betaConfounding,betaNeighborConfounding)

    epsilonTrain,epsilonVal,epsilonTest = noiseSimulation(data,trainIndex,valIndex,testIndex)

    w_z2 = 2 * np.random.random_sample((dimension)) - 1 #effect of T to Y

    POTrain = potentialOutcomeSimulation(w_z2,trainX,trainA,T_train,epsilonTrain,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise)
    POVal = potentialOutcomeSimulation(w_z2,valX,valA,T_val,epsilonVal,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise)
    POTest = potentialOutcomeSimulation(w_z2,testX,testA,T_test,epsilonTest,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise)



    cfT_train,nodesToFlipTrain = flipTreatment(T_train,flipRate)
    cfT_val,nodesToFlipVal = flipTreatment(T_val,flipRate)
    cfT_test,nodesToFlipTest = flipTreatment(T_test,flipRate)

    epsilonTrain,epsilonVal,epsilonTest = noiseSimulation(data,trainIndex,valIndex,testIndex)

    cfPOTrain = potentialOutcomeSimulation(w_z2,trainX,trainA,cfT_train,epsilonTrain,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise)
    cfPOVal = potentialOutcomeSimulation(w_z2,valX,valA,cfT_val,epsilonVal,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise)
    cfPOTest = potentialOutcomeSimulation(w_z2,testX,testA,cfT_test,epsilonTest,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise)



    num = trainX.shape[0]
    t_1s = np.ones(num)
    t_0s = np.zeros(num)
    z_7s = np.zeros(num)+z_11
    z_2s = np.zeros(num)+z_22
    z_1s = np.ones(num)
    z_0s = np.zeros(num)


    cfPOTrain_t1z1 = potentialOutcomeSimulation(w_z2,trainX,trainA,t_1s,epsilonTrain,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_1s)
    cfPOTrain_t1z0 = potentialOutcomeSimulation(w_z2,trainX,trainA,t_1s,epsilonTrain,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_0s)
    cfPOTrain_t0z0 = potentialOutcomeSimulation(w_z2,trainX,trainA,t_0s,epsilonTrain,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_0s)
    cfPOTrain_t0z7 = potentialOutcomeSimulation(w_z2,trainX,trainA,t_0s,epsilonTrain,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_7s)
    cfPOTrain_t0z2 = potentialOutcomeSimulation(w_z2,trainX,trainA,t_0s,epsilonTrain,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_2s)

    num = valX.shape[0]
    t_1s = np.ones(num)
    t_0s = np.zeros(num)
    z_7s = np.zeros(num)+z_11
    z_2s = np.zeros(num)+z_22
    z_1s = np.ones(num)
    z_0s = np.zeros(num)
    cfPOVal_t1z1 = potentialOutcomeSimulation(w_z2,valX,valA,t_1s,epsilonVal,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_1s)
    cfPOVal_t1z0 = potentialOutcomeSimulation(w_z2,valX,valA,t_1s,epsilonVal,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_0s)
    cfPOVal_t0z0 = potentialOutcomeSimulation(w_z2,valX,valA,t_0s,epsilonVal,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_0s)
    cfPOVal_t0z7 = potentialOutcomeSimulation(w_z2,valX,valA,t_0s,epsilonVal,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_7s)
    cfPOVal_t0z2 = potentialOutcomeSimulation(w_z2,valX,valA,t_0s,epsilonVal,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_2s)


    num = testX.shape[0]
    t_1s = np.ones(num)
    t_0s = np.zeros(num)
    z_7s = np.zeros(num)+z_11
    z_2s = np.zeros(num)+z_22
    z_1s = np.ones(num)
    z_0s = np.zeros(num)
    cfPOTest_t1z1 = potentialOutcomeSimulation(w_z2,testX,testA,t_1s,epsilonTest,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_1s)
    cfPOTest_t1z0 = potentialOutcomeSimulation(w_z2,testX,testA,t_1s,epsilonTest,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_0s)
    cfPOTest_t0z0 = potentialOutcomeSimulation(w_z2,testX,testA,t_0s,epsilonTest,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_0s)
    cfPOTest_t0z7 = potentialOutcomeSimulation(w_z2,testX,testA,t_0s,epsilonTest,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_7s)
    cfPOTest_t0z2 = potentialOutcomeSimulation(w_z2,testX,testA,t_0s,epsilonTest,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_2s)



    train = {'T':np.array(T_train),
            'cfT':np.array(cfT_train),
            'features': trainX, 
            'PO':POTrain,
            'cfPO':cfPOTrain,
            'nodesToFlip':nodesToFlipTrain,
            'network':trainA,
            "meanT":meanT_train,

            "train_t1z1":cfPOTrain_t1z1,
            "train_t1z0":cfPOTrain_t1z0,
            "train_t0z0":cfPOTrain_t0z0,
            "train_t0z7":cfPOTrain_t0z7,
            "train_t0z2":cfPOTrain_t0z2,
    }

    val = {'T':np.array(T_val),
            'cfT':np.array(cfT_val), 
            'features': valX, 
            'PO':POVal,
            'cfPO':cfPOVal,
            'nodesToFlip':nodesToFlipVal,
            'network':valA,
            "meanT":meanT_val,
            
            "val_t1z1":cfPOVal_t1z1,
            "val_t1z0":cfPOVal_t1z0,
            "val_t0z0":cfPOVal_t0z0,
            "val_t0z7":cfPOVal_t0z7,
            "val_t0z2":cfPOVal_t0z2}

    test = {'T':np.array(T_test), 
            'cfT':np.array(cfT_test),
            'features': testX, 
            'PO':POTest,
            'cfPO':cfPOTest,
            'nodesToFlip':nodesToFlipTest,
            'network':testA,
            "meanT":meanT_test,
            
            "test_t1z1":cfPOTest_t1z1,
            "test_t1z0":cfPOTest_t1z0,
            "test_t0z0":cfPOTest_t0z0,
            "test_t0z7":cfPOTest_t0z7,
            "test_t0z2":cfPOTest_t0z2,
            }

    data = {"train":train,"val":val,"test":test}

    saveData(dataset,data,expID,flipRate)



    print ("***************************************************************************************")
    print ("Dataset:{},expID:{},flipRate:{} is Done!!".format(dataset,expID,flipRate))
    print ("***************************************************************************************")




if __name__=="__main__":
    datasets = ["BC","Flickr"]
    dimension = 10
    flipRates = [0.25,0.5,0.75,1]
    expIDs = [0,1,2,3,4]
    z_11 = 0.7
    z_22 = 0.2
    for dataset in datasets:
        for flipRate in flipRates:
                for expID in expIDs:
                    main(dataset,dimension,flipRate,expID,z_11,z_22)




        






