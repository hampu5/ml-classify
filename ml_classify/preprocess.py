#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Some preprocessing code- preprocess to generate time and 
frequency-based features using a window sizing technique 
(Rosell) and test random forest on it

'''
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import cross_val_score


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score


from sklearn.feature_selection import SelectFromModel

import time


import os
import glob
import pandas as pd



from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
#from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score,roc_curve

import time


from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.feature_selection import *
from sklearn.metrics import *

#from matplotlib import pyplot as plt
#import seaborn as sns
#Might have some redundant imports

#%%
datasets = {}


# ORNL DATA
PATH = "/home/hampus/miun/master_thesis/Datasets/ORNL/"

ambient = pd.read_csv( f"{PATH}/ambient.csv" )
attack = pd.read_csv( f"{PATH}/attack.csv" )

ambient["filename"] = ambient["filename"].apply( lambda x: PATH + x)
attack["filename"] = attack["filename"].apply( lambda x: PATH + x)


ambient["has_attacks"] = False
attack["has_attacks"] = True

ambient = ambient[ ["name", "filename", "has_attacks"] ]
attack =  attack[ ["name", "filename", "has_attacks"] ]

df1 = pd.concat( [ ambient, attack])
df1["remarks"] = "No DLC available"
datasets["ORNL"] = df1.to_dict("records")



#%%
#Features related to time

# various functions to get the data into the format we want
def calc_relative_time(AT):    
    """helper function to translate absolute time to relative time"""
    
    dt = AT.diff()
    dt[0] = dt.mean() # fill the missing first element
    return dt


def calc_relative_time_per_id(df):
    
    df["dt_ID"] = df.groupby(by = "ID")["t"].diff()
    

    nans = df["dt_ID"].index[df["dt_ID"].apply(np.isnan)]

    nans_idx = df["dt_ID"].index[df["dt_ID"].apply(np.isnan)]
    nans_ids = [int(df.iloc[d]["ID"]) for d in nans_idx ]

    meanall = df["dt_ID"].mean() # needed when an ID is used only once, hence no mean
    means_ = df.groupby(by = "ID")["dt_ID"].mean().fillna(meanall).to_dict() # mean for each ID
    nans_vals = [ means_[id_] for id_ in nans_ids]
    

    tmp = df["dt_ID"].copy()
    tmp.iloc[nans_idx] = nans_vals
    df["dt_ID"] = tmp

    assert df.dt_ID.isnull().sum() == 0

#%%
def read_file(filename):
    df = pd.read_csv(filename)
    
    if "Timestamp" in df.columns:
        df.rename(columns = {'Timestamp':'t'}, inplace = True)
    
    df["dt"] =  calc_relative_time(df["t"])
    calc_relative_time_per_id(df)
    
    return df

#%%


def get_labels_by_class(class_, window_size): 
    # return [c for c in df.columns if c.startswith(class_ + "_")]    
    return [ f"{class_}_{i}" for i in range(window_size) ]

#%%
#Features related to frequency using window
def extract_window(df, window_size):

    cs = df.columns
    dfw = pd.DataFrame()
    id_column_names =[] #added
    for i in range(window_size):
        tmp = df.shift(-i)
        column_names = {c: f"{c}_{i}" for c in cs}
        id_column_names.append(column_names['ID']) #added
        tmp = tmp.rename(columns = column_names)
        if dfw.shape[0] == 0:
            dfw = tmp
        else:
            dfw = dfw.join(tmp)
    
    print('ID NAMES: %s' % id_column_names) #added

    

    ## Calculate 
    from collections import Counter #added
    dfw["win_most_freq_id"] = dfw.apply(lambda x: list([x[name] for name in id_column_names]), axis=1) #added
    dfw["win_most_freq_id"]= dfw["win_most_freq_id"].apply(lambda x: max(Counter(x).values())) #added


    
    # drop the last ones
    len_ = dfw.shape[0]
    dfw.drop( range(len_ - window_size, len_), inplace = True)

    # compute the labels and remove the individual labels    
    lbs = get_labels_by_class("Label", window_size)    
    dfw["Label"] = dfw[lbs].max( axis = 1)
    dfw.drop(lbs, axis = 1, inplace = True)
    
    # compute win_dir and win_mean_dt:
    dts = get_labels_by_class("dt", window_size)    
    dfw["win_dur"] = dfw[dts].sum( axis = 1)
    dfw["win_mean_dt"] = dfw[dts].mean( axis = 1)
    
    # drop the absolut times, they are useless now
    ts = get_labels_by_class("t", window_size)    
    dfw.drop(ts, axis = 1, inplace = True)
    
    
    Y = dfw["Label"]
    X = dfw.drop("Label", axis = 1)
    return X, Y


#%%
df_attack=pd.DataFrame()
df_ambient=pd.DataFrame()
#%%
for dname, dataset in datasets.items():
    for dataitem in dataset:
        name = dataitem["name"]
        filename = dataitem["filename"]
        has_attacks = bool( dataitem["has_attacks"] )
        remarks = dataitem["remarks"] or ""
        
         
        if has_attacks:             
            df = read_file(filename)
            df_attack = df_attack.append(df, ignore_index=True)
        else:             
            df = read_file(filename)
            df_ambient = df_ambient.append(df, ignore_index=True)    
                  

#%%
#print(df_attack)
#print(df_ambient)

#%%

#attack_Y=df_attack["Label"]
#attack_X=df_attack.drop(["Label", "ID"], axis=1)

#%%
window_size=10
attack_X,attack_Y =extract_window(df_attack, window_size)

print(attack_X.shape)
#%%
#attack_X.head()
#%%
#window_size=10
#df_ambient,ambient_Y =extract_window(df_ambient, window_size)
#print(df_ambient.shape)
#%%
#print(df_ambient.head())

#%%
# feature selection:
features=100
selector = SelectKBest(chi2, k = features)
print("KBest Selected")

#%%
X_train, X_val, y_train, y_val = train_test_split(attack_X, attack_Y,test_size=0.5, random_state=2, shuffle=True)

x_train = selector.fit_transform(X_train, y_train)

#%%
x_test = selector.fit_transform(X_val,y_val)
print(x_train.shape)

#%%
#Classification with Random Forest

clf=RandomForestClassifier(n_estimators=100)
#%%

clf.fit(x_train, y_train)

#%%
scores = cross_val_score(clf, x_train, y_train, scoring='f1', cv=10)
print("Training F1: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
#%%
pred = clf.predict(x_test)

#%%
f1_scores = f1_score(y_val, pred, average='weighted')
print("Testing F1:  %0.4f(+/- %0.4f)" % (f1_scores.mean(), f1_scores.std()))
   