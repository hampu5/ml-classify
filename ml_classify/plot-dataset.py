from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns


datasets = {}


# ORNL DATA
PATH_ORNL = "/home/hampus/projects/Datasets/ORNL/"
PATH_SURVIVAL = "/home/hampus/projects/Datasets/Survival/"
PATH_HISINGEN = "/home/hampus/projects/Datasets/Hisingen/"


def load_dataset(path, filename, has_attacks):
    data = pd.read_csv(f"{path}/{filename}")
    data["filename"] = data["filename"].apply(lambda x: path + x)
    data["has_attacks"] = has_attacks
    data = data[["name", "filename", "has_attacks"]]
    return data


# various functions to get the data into the format we want
def calc_relative_time(AT):    
    # helper function to translate absolute time to relative time
    dt = AT.diff()
    dt[0] = dt.mean() # fill the missing first element
    return dt

def calc_relative_time_per_id(df):
    df["dt_ID"] = df.groupby(by="ID")["t"].diff()

    nans_idx = df["dt_ID"].index[df["dt_ID"].apply(np.isnan)]
    nans_ids = [int(df.iloc[d]["ID"]) for d in nans_idx]

    meanall = df["dt_ID"].mean() # needed when an ID is used only once, hence no mean
    means_ = df.groupby(by="ID")["dt_ID"].mean().fillna(meanall).to_dict() # mean for each ID
    nans_vals = [means_[id_] for id_ in nans_ids]
    
    tmp = df["dt_ID"].copy()
    tmp.iloc[nans_idx] = nans_vals
    df["dt_ID"] = tmp

    assert df.dt_ID.isnull().sum() == 0

def read_file(filename):
    df = pd.read_csv(filename)
    
    if "Timestamp" in df.columns:
        df.rename(columns={'Timestamp':'t'}, inplace=True)
    
    df["dt"] = calc_relative_time(df["t"])
    calc_relative_time_per_id(df)
    
    return df


# ambient = load_dataset(PATH_ORNL, "ambient.csv", False)
# attack = load_dataset(PATH_ORNL, "attack.csv", True)

# df1 = pd.concat([ambient, attack])
# df1["remarks"] = "No DLC available"
# datasets["ORNL"] = df1.to_dict("records")

# # Release memory
# ambient = None
# attack = None

df1 = load_dataset(PATH_SURVIVAL, "data.csv", True)
df1["remarks"] = "-"
datasets["Survival"] = df1.to_dict("records")

# df1 = load_dataset(PATH_HISINGEN, "data.csv", True)
# df1["remarks"] = "-"
# datasets["Survival"] = df1.to_dict("records")

# Release memory
df1 = None


df_attack = pd.DataFrame()
df_ambient = pd.DataFrame()

for dname, dataset in datasets.items():
    for dataitem in dataset:
        name = dataitem["name"]
        filename = dataitem["filename"]
        has_attacks = bool(dataitem["has_attacks"])
        remarks = dataitem["remarks"] or ""
        
        if has_attacks:             
            df = read_file(filename)
            df_attack = pd.concat([df_attack, df], ignore_index=True)
        # else:             
        #     df = read_file(filename)
        #     df_ambient = pd.concat([df_ambient, df], ignore_index=True)

df_ambient = None # Release memory, as it isn't used for now


df_attack.drop(["t", "ID", "DLC", "dt", "dt_ID"], axis=1, inplace=True)

#df_attack = df_attack[:10000]
print(df_attack)

sns.pairplot(df_attack, corner=True, hue="Label")


# plot correlation matrix
# names = df_attack.columns
# correlations = df_attack.corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0, names.size, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)

# df_attack.plot(kind='density', subplots=True, sharex=False)
plt.show()
