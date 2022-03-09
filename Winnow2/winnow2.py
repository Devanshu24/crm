#!/usr/bin/env python
# coding: utf-8

# **Winnow2 Algorithm**
#
# *Author: Tirtharaj Dash, BITS Pilani, Goa Campus ([Homepage](https://tirtharajdash.github.io))*

import pickle

import numpy as np
import pandas as pd

# from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Use seed for reproducibility of results
seedval = 2018
np.random.seed(seedval)


# function to load data file and return train, val, test split
def load_data(trainfile="train.csv", testfile="test.csv"):

    # load the data and prepare X and y
    Data = pd.read_csv("train.csv", header=None)
    print(
        "Dimension of Data ( Instances: ",
        Data.shape[0],
        ", Features: ",
        Data.shape[1] - 1,
        " )",
    )

    # X = Data[Data.columns[0:3800]] #only for Hide-and-Seek features to match ILP results
    X = Data.drop([Data.columns[-1]], axis=1)  # for random features
    y = Data[Data.columns[-1]]
    print("Dimension of X: ", X.shape)
    print("Dimension of y: ", y.shape)

    # prepare the training and validation set from X and y
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
    )

    print("\nDimension of X_train: ", X_train.shape)
    print("Dimension of y_train: ", y_train.shape)
    print("Dimension of X_val: ", X_val.shape)
    print("Dimension of y_val: ", y_val.shape)

    # load the holdout/test set
    Data_test = pd.read_csv("test.csv", header=None)
    print(
        "\nDimension of Data_test ( Instances: ",
        Data_test.shape[0],
        ", Features: ",
        Data_test.shape[1] - 1,
        " )",
    )

    # X_test = Data_test[Data_test.columns[0:3800]]
    X_test = Data_test.drop([Data_test.columns[-1]], axis=1)  # for random features
    y_test = Data_test[Data_test.columns[-1]]
    print("Dimension of X: ", X_test.shape)
    print("Dimension of y: ", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


# function to calculate accuracy
def accuracy_score(y, y_pred):

    acc = np.mean(y == y_pred)
    return acc


# function to calculate netsum and prediction for one instance
def predictOne(W, X, thres):

    netsum = np.sum(W * X)  # net sum

    # threshold check
    if netsum >= thres:
        y_hat = 1
    else:
        y_hat = 0

    return y_hat


# function to calculate netsums and predictions for all instances
def predictAll(W, X, thres):

    NetSum = np.dot(X, W)
    Idx = np.where(NetSum >= thres)
    y_pred = np.zeros(X.shape[0])
    y_pred[Idx] = 1

    return y_pred


# function to compute and print the classification summary
def ComputePerf(W, X, y, thres, print_flag=False):

    y_pred = predictAll(W, X, thres)  # compute the prediction
    acc = accuracy_score(y, y_pred)  # compute accuracy

    # print the summary if printflag is True
    if print_flag:
        print("Accuracy:", acc)
        # print(confusion_matrix(y, y_pred))
        # print(classification_report(y, y_pred))

    return acc


# function to train Winnow2 using grid-search (optional)
def TrainWinnow2(
    X_train, y_train, X_val, y_val, params, max_epoch=10, patience=20, verbose=False
):

    n = X_train.shape[1]
    best_perf = 0.0  # best val set performance so far
    grid_space = len(params["Alpha"]) * len(params["Thres"])  # size of grid space
    grid_iter = 0  # grid search iterator

    # model dictionary
    model = {"W": [], "alpha": None, "thres": None, "train_perf": 0.0, "val_perf": 0.0}

    # grid search and training
    for alpha in params["Alpha"]:

        for thres in params["Thres"]:

            grid_iter += 1

            print("-----------------------------------------------------------------")
            print(
                "Trying::\t alpha:",
                alpha,
                "threshold:",
                thres,
                "\t(",
                grid_iter,
                "of",
                grid_space,
                ")",
            )
            print("-----------------------------------------------------------------")

            W = np.ones(n)  # winnow2 initialisation
            piter = 0  # patience iteration
            modelfound = False  # model found flag

            for epoch in range(0, max_epoch):

                # Winnow loop (computation and update) starts
                for i in range(1, X_train.shape[0]):

                    y_hat = predictOne(W, X_train.iloc[i, :], thres)

                    # Winnow prediction is a mismatch
                    if y_train.iloc[i] != y_hat:

                        # active attribute indices
                        Idx = np.where(X_train.iloc[i, :] == 1)

                        if y_hat == 0:
                            W[Idx] *= alpha  # netsum was too small (promotion step)

                        else:
                            W[Idx] /= alpha  # netsum was too high (demotion step)

                # compute performance on val set
                val_perf = ComputePerf(W, X_val, y_val, thres)

                if verbose:
                    train_perf = ComputePerf(W, X_train, y_train, thres)
                    print(
                        "[Epoch %d] train_perf: %6.4f \tval_perf: %6.4f"
                        % (epoch, train_perf, val_perf)
                    )

                # is it a better model
                if val_perf > best_perf:
                    best_perf = val_perf
                    piter = 0  # reset the patience count
                    modelfound = True  # model is found
                    train_perf = ComputePerf(W, X_train, y_train, thres)

                    # update the model with new params
                    model["W"] = W.copy()  # optima W
                    model["alpha"] = alpha  # optimal alpha
                    model["thres"] = thres  # optimal threshold
                    model["train_perf"] = train_perf  # training performance
                    model["val_perf"] = val_perf  # validation performance

                    print(
                        "[Selected] at epoch",
                        epoch,
                        "\ttrain_perf: %6.4f \tval_perf: %6.4f"
                        % (train_perf, val_perf),
                    )

                else:
                    piter = piter + 1

                if piter >= patience:
                    print("Stopping early after epoch", (epoch + 1))
                    break

            if not modelfound:
                print("No better model found.\n")
            else:
                print("Model found and saved.\n")

    return model


# main function
def main():

    # call function to get the splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        trainfile="train.csv", testfile="test.csv"
    )

    # Winnow2 algorithm hyperparameters (will be decided based on a validation set)
    n = X_train.shape[1]  # number of features
    Thres = [n, n // 2]  # decision threshold
    Alpha = [2, 3, 4]  # multiplicative factor for weight (promotion or demotion)

    # algorithm param dict
    params = {"Thres": Thres, "Alpha": Alpha}

    # trainign hyperparameters
    patience = 20  # patience period for early-stopping # noqa
    max_epoch = 200  # maximum Winnow epochs

    # call for training and model selection
    model = TrainWinnow2(
        X_train,
        y_train,
        X_val,
        y_val,
        params=params,
        max_epoch=max_epoch,
        patience=20,
        verbose=False,
    )

    # Optimal hyperparameters for Winnow2 using validation set
    print("Best model:", model)

    # test the performance of model on test set
    acc = ComputePerf(model["W"], X_test, y_test, model["thres"])
    print("Independent test accuracy:", acc)

    # store the results
    with open("score.txt", "w") as fp:
        fp.write(str(acc) + "\n")
    fp.close()

    # save the trained model (dict)
    with open("model.pkl", "wb") as fp:
        pickle.dump(model, fp, pickle.HIGHEST_PROTOCOL)
    fp.close()

    # load and test the saved model
    # with open('model.pkl', 'rb') as fp:
    #    savedmodel = pickle.load(fp)
    # acc = ComputePerf(savedmodel['W'], X_test, y_test, savedmodel['thres'])
    # print(acc)
    # print(classification_report(y_test, predictAll(savedmodel['W'], X_test, savedmodel['thres'])))
    # fp.close()


# for analysis of the weights
# with open('model.pkl', 'rb') as fp:
#    savedmodel = pickle.load(fp)
# df = pd.DataFrame(savedmodel['W'])
# df.to_csv("weights.csv")


if __name__ == "__main__":
    main()
