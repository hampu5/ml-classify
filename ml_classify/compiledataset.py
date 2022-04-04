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
    data = data[["name", "class", "type", "filename", "has_attack"]]
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
    # drop_bytes(df)

    df_data = df_data.applymap(format_binary)
    
    df["data"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    
    df_data = None
    
    assert not df["data"].isnull().values.any(axis=None)

def dec_payload(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    # drop_bytes(df)
    
    df["data_dec"] = \
        df_data["d0"] * (256**7) + \
        df_data["d1"] * (256**6) + \
        df_data["d2"] * (256**5) + \
        df_data["d3"] * (256**4) + \
        df_data["d4"] * (256**3) + \
        df_data["d5"] * (256**2) + \
        df_data["d6"] * 256 + \
        df_data["d7"]
    
    df_data = None
    
    assert not df["data_dec"].isnull().values.any(axis=None)

def hex_payload(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    # drop_bytes(df)

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
    print(f"number of nans in dt_ID: {len(nans_idx)}")
    # print(nans_idx)
    nans_ids = [int(df.iloc[d]["ID"]) for d in nans_idx]

    meanall = df["dt_ID"].mean() # needed when an ID is used only once, hence no mean
    means_ = df.groupby(by="ID")["dt_ID"].mean().fillna(meanall).to_dict() # mean for each ID
    nans_vals = [means_[id_] for id_ in nans_ids]
    
    tmp = df["dt_ID"].copy()
    tmp.iloc[nans_idx] = nans_vals
    df["dt_ID"] = tmp

    assert no_nan_or_inf(df["dt_ID"])

def create_dt_data(df: pd.DataFrame):
    df["dt_data"] = df.groupby(by="data_dec")["t"].diff()

    df["dt_data"].fillna(df["dt_data"].mean(), inplace=True)

    # nans_idx = df["dt_data"].index[df["dt_data"].apply(np.isnan)]
    # print(f"number of nans in dt_data: {len(nans_idx)}")
    # # print(nans_idx)
    # nans_data = [int(df.iloc[d]["data_dec"]) for d in nans_idx]

    # meanall = df["dt_data"].mean() # needed when a Data is used only once, hence no mean
    # means_ = df.groupby(by="data_dec")["dt_data"].mean().fillna(meanall).to_dict() # mean for each Data
    # nans_vals = [means_[id_] for id_ in nans_data]
    
    # tmp = df["dt_data"].copy()
    # tmp.iloc[nans_idx] = nans_vals
    # df["dt_data"] = tmp

    assert no_nan_or_inf(df["dt_data"])

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


def create_dc(df: pd.DataFrame):
    df["dc"] = df.groupby("ID")["data"].shift().ne(df['data']).astype(int)

    assert no_nan_or_inf(df["dc"])


def smc(s1, s2): # Simple Matching Coefficient
    return sum([1 for c1, c2 in zip(s1, s2) if c1 == c2]) / 64

def check_unique(x):
    shifted = x.shift(1)
    temp = {}
    for (idx1, s1), (idx2, s2) in zip(x.items(), shifted.items()):
        if not (isinstance(s1, str) and isinstance(s2, str)):
            temp[idx2] = np.nan
            continue
        temp[idx2] = 1 - smc(s1, s2)
    return pd.Series(temp)

def create_dcs(df: pd.DataFrame):
    dcs = df.groupby("ID")["data"].apply(check_unique)
    df["dcs"] = dcs.fillna(dcs.mean())

    assert no_nan_or_inf(df["dcs"])


def read_file(filename):
    df = pd.read_csv(filename)
    
    df.rename(columns={'Timestamp':'t'}, inplace=True, errors="ignore")
    
    binary_payload(df)
    dec_payload(df)
    drop_bytes(df)

    create_dt(df)
    create_dt_ID(df)
    create_dt_data(df)
    # create_dt_ID_data(df)
    # create_dc(df)
    create_dcs(df)
    
    return df



def compile_dataset(datasets: dict):
    df_attack = pd.DataFrame()
    df_ambient = pd.DataFrame()

    for dname, dataset in datasets.items():
        for dataitem in dataset:
            name = dataitem["name"]
            c = dataitem["class"]
            atype = dataitem["type"]
            filename = dataitem["filename"]
            has_attack = bool(dataitem["has_attack"])
            # remarks = dataitem["remarks"] or ""
            df = read_file(filename)
            df["name"] = name
            df["class"] = c
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