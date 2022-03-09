#!/usr/bin/env python
# coding: utf-8

# **Winnow2 Algorithm**
#
# *Author: Tirtharaj Dash, BITS Pilani, Goa Campus ([Homepage](https://tirtharajdash.github.io))*

import pickle

import pandas as pd
import winnow2
from sklearn.metrics import classification_report

with open("./Results/chess3/model.pkl", "rb") as fp:
    savedmodel = pickle.load(fp)

Data = pd.read_csv("test3.pos.csv", header=None)
X = Data.drop([Data.columns[-1]], axis=1)
y = Data[Data.columns[-1]]

acc = winnow2.ComputePerf(savedmodel["W"], X, y, savedmodel["thres"])
y_pred = winnow2.predictAll(savedmodel["W"], X, savedmodel["thres"])
print(classification_report(y, y_pred))
