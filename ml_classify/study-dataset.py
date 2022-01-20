import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_selection import SelectKBest, chi2

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

def read_file(filename):
    df = pd.read_csv(filename)
    
    if "Timestamp" in df.columns:
        df.rename(columns = {'Timestamp':'t'}, inplace = True)
    
    df["dt"] =  calc_relative_time(df["t"])
    calc_relative_time_per_id(df)
    
    return df


df_attack=pd.DataFrame()
df_ambient=pd.DataFrame()

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
                  


# print(df_ambient)
# print(df_attack)



def get_labels_by_class(class_, window_size): 
    # return [c for c in df.columns if c.startswith(class_ + "_")]    
    return [ f"{class_}_{i}" for i in range(window_size) ]

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


window_size=10
attack_X,attack_Y = extract_window(df_attack, window_size)
print(attack_X.shape)


# feature selection:
features=100
selector = SelectKBest(chi2, k = features)
print("KBest Selected")