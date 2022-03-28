from math import ceil
import string
import numpy as np
import pandas as pd
import re

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
import keras_tuner as kt

# Building the model with Keras Tuner

def build_model(hp):
    model = keras.Sequential()
    model.add(Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=4, max_value=10, step=1),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(Dropout(rate=0.25))
    model.add(Dense(1, activation="sigmoid"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)


# Loading different datasets
def load_dataset(path, filename) -> pd.DataFrame:
    data = pd.read_csv(f"{path}/{filename}")
    data["filename"] = data["filename"].apply(lambda filename: f"{path}/{filename}")
    data = data[["name", "type", "filename", "has_attack"]]
    return data


# Helpers

def format_binary(val):
    return f"{val:08b}"

def format_hex(val):
    return f"{val:02x}"

def no_nan_or_inf(s: pd.Series):
    s.replace([np.inf, -np.inf], np.nan, inplace=True)
    return not s.isnull().values.any(axis=None)

def drop_bytes(df: pd.DataFrame):
    df.drop(columns=["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"], inplace=True, errors="ignore")

def binary_payload(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    drop_bytes(df)

    df_data = df_data.applymap(format_binary)
    
    df["data"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    
    df_data = None
    
    assert not df["data"].isnull().values.any(axis=None)

def hex_payload(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    drop_bytes(df)

    df_data = df_data.applymap(format_hex)
    
    df["data"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    
    df_data = None
    
    assert not df["data"].isnull().values.any(axis=None)

# Various functions to get the data into the format we want

# Helper function to translate absolute time to relative time
def create_dt(df: pd.DataFrame):
    dt = df["t"].diff()
    dt.iloc[0] = dt.mean() # fill the missing first element
    df["dt"] = dt

    assert no_nan_or_inf(df["dt"])

# Calculate relative time between IDs
def create_dt_ID(df: pd.DataFrame):
    df["dt_ID"] = df.groupby(by="ID")["t"].diff()

    nans_idx = df["dt_ID"].index[df["dt_ID"].apply(np.isnan)]
    # print(nans_idx)
    nans_ids = [int(df.iloc[d]["ID"]) for d in nans_idx]

    meanall = df["dt_ID"].mean() # needed when an ID is used only once, hence no mean
    means_ = df.groupby(by="ID")["dt_ID"].mean().fillna(meanall).to_dict() # mean for each ID
    nans_vals = [means_[id_] for id_ in nans_ids]
    
    tmp = df["dt_ID"].copy()
    tmp.iloc[nans_idx] = nans_vals
    df["dt_ID"] = tmp

    assert no_nan_or_inf(df["dt_ID"])

def create_dt_ID_data_bytewise(df: pd.DataFrame):
    for i in range(8):
        df[f"dt_ID_d{i}"] = df.groupby(by=["ID", f"d{i}"])["t"].diff()

        nans_idx = df[f"dt_ID_d{i}"].index[df[f"dt_ID_d{i}"].apply(np.isnan)]
        # print(nans_idx)
        nans_ids = [int(df.iloc[d]["ID"]) for d in nans_idx]

        meanall = df[f"dt_ID_d{i}"].mean() # needed when an ID is used only once, hence no mean
        means_ = df.groupby(by="ID")[f"dt_ID_d{i}"].mean().fillna(meanall).to_dict() # mean for each ID
        nans_vals = [means_[id_] for id_ in nans_ids]
        
        tmp = df[f"dt_ID_d{i}"].copy()
        tmp.iloc[nans_idx] = nans_vals
        df[f"dt_ID_d{i}"] = tmp

        assert no_nan_or_inf(df[f"dt_ID_d{i}"])

def create_dt_ID_data(df: pd.DataFrame):
    df["dt_ID_data"] = df.groupby(by=["ID", "data"])["t"].diff()

    nans_idx = df["dt_ID_data"].index[df["dt_ID_data"].apply(np.isnan)]
    # print(nans_idx)
    nans_ids = [int(df.iloc[d]["ID"]) for d in nans_idx]

    meanall = df["dt_ID_data"].mean() # needed when an ID is used only once, hence no mean
    means_ = df.groupby(by="ID")["dt_ID_data"].mean().fillna(meanall).to_dict() # mean for each ID
    nans_vals = [means_[id_] for id_ in nans_ids]
    
    tmp = df["dt_ID_data"].copy()
    tmp.iloc[nans_idx] = nans_vals
    df["dt_ID_data"] = tmp

    assert no_nan_or_inf(df["dt_ID_data"])

from IPython.display import display

# def check_unique(x):
#     shifted = x.shift(1)
#     return (x != shifted).mask(shifted.isna())

def create_dc(df: pd.DataFrame):
    # dc = df.groupby("ID")["data"].apply(check_unique).fillna(True).astype(int)
    # display(dc)
    # df["dc"] = dc
    df["dc"] = df.groupby("ID")["data"].shift().ne(df['data']).astype(int)

    assert no_nan_or_inf(df["dc"])


def check_unique(x):
    shifted = x.shift(1)
    temp = []
    for s1, s2 in zip(x, shifted):
        print(s1)
        print(s2)
        if s1 == np.nan or s2 == np.nan:
            temp.append(np.nan)
            continue
        temp.append(sum([1 for c1, c2 in zip(s1, s2) if c1 == c2]) / 64)
    return pd.Series(temp)

def create_dcs(df: pd.DataFrame):
    dcs = df.groupby("ID")["data"].apply(check_unique)
    display(dcs)
    df["dcs"] = dcs
    # df["dc"] = df.groupby("ID")["data"].shift().ne(df['data']).astype(int)

    assert no_nan_or_inf(df["dcs"])

# grouped = pd.DataFrame([[0, 1], [0, 1], [1, 0]], columns=["A", "B"]).groupby("A")
# for name, group in grouped:
#     shifted = group.shift(1)
    
# display(result)

# Similarity coefficient between two string
# def smc(s1: str, s2: str):
#     return sum([1 for c1, c2 in zip(s1, s2) if c1 == c2]) / 64

# def change_score(s):
#     shifted = s.shift(1)
#     return smc(s, shifted)

# from sklearn.metrics import pairwise_distances
# def create_dcs(df: pd.DataFrame):
#     df["dcs"] = 0
#     grouped = df.groupby("ID")
#     for name, group in grouped:
#         if len(group) < 2:
#             idx = group.index[i]
#             df["dcs"].iloc[idx] = np.nan
#             continue
#         for i, row in enumerate(group, 1):
#             idx = group.index[i-1]
#             print(idx)
#             df["dcs"].iloc[idx] = smc(group.iloc[i]["data"], group.iloc[i-1]["data"])

    # assert no_nan_or_inf(df["dcs"])

# print(int("str1",2) and int("str2",2))

def read_file(filename):
    df = pd.read_csv(filename)
    
    df.rename(columns={'Timestamp':'t'}, inplace=True, errors="ignore")
    
    binary_payload(df)

    create_dt(df)
    create_dt_ID(df)
    create_dt_ID_data(df)
    # create_dc(df)
    create_dcs(df)
    
    return df



def compile_dataset(datasets: dict):
    df_attack = pd.DataFrame()
    df_ambient = pd.DataFrame()

    for dname, dataset in datasets.items():
        for dataitem in dataset:
            name = dataitem["name"]
            atype = dataitem["type"]
            filename = dataitem["filename"]
            has_attack = bool(dataitem["has_attack"])
            # remarks = dataitem["remarks"] or ""
            df = read_file(filename)
            df["name"] = name
            df["dataset"] = dname
            df["type"] = "none"
            df.loc[df["Label"] == 1, "type"] = atype
            # print(df["type"])
            if has_attack:
                df_attack = pd.concat([df_attack, df], ignore_index=True)
            else:
                df_ambient = pd.concat([df_ambient, df], ignore_index=True)
    
    df_all = pd.concat([df_attack, df_ambient], ignore_index=True)
    # df_all = feature_creation(df_all)
    df_all = df_all[[c for c in df_all if c not in ["dataset", "type", "Label"]] + ["dataset", "type", "Label"]]

    # assert not df_all.isnull().values.any(axis=None)

    return df_all