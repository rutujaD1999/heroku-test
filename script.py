import datetime
import numpy as np
from matplotlib import cm, pyplot as plt
import pandas as pd
from pandas_datareader import data
import seaborn as sns
from sklearn import tree
import talib

duration = 7
df = data.DataReader("GAIL.NS", start='2014-1-1', end='2020-12-06', data_source='yahoo')
df.to_csv("SBIN.csv")
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
df=pd.read_csv("SBIN.csv")
df.tail()
