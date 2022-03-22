from math import ceil
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

def create_dt_ID_data(df: pd.DataFrame):
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


def read_file(filename):
    df = pd.read_csv(filename)
    
    df.rename(columns={'Timestamp':'t'}, inplace=True, errors="ignore")
    
    create_dt(df)
    create_dt_ID(df)
    create_dt_ID_data(df)
    
    return df


def get_binary_payload(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    drop_bytes(df)

    df_data = df_data.applymap(format_binary)
    
    df["bin_data"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    
    df_data = None
    
    assert not df["bin_data"].isnull().values.any(axis=None)

def get_hex_payload(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    drop_bytes(df)

    df_data = df_data.applymap(format_hex)
    
    df["hex_data"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    
    df_data = None
    
    assert not df["hex_data"].isnull().values.any(axis=None)


def new_feature(df: pd.DataFrame):
    # count_ones_weighted2(df)
    # count_ones_weighted(df)
    # count_ones(df)
    # count_runs(df)
    # count_runs_diff(df)
    # df = merge_data_features(df)
    # get_binary_payload(df)
    # get_hex_payload(df)


    
    # drop_bytes(df)

    # assert not df.isnull().values.any(axis=None)
    
    return df

def feature_creation(df: pd.DataFrame):
    # size = len(df.index)
    # number_of_splits = ceil(size / 5000000)
    # splits = np.array_split(df, number_of_splits)
    # df = pd.DataFrame

    # for split in splits:
    #     if df.empty:
    #         df = new_feature(split)
    #     else:
    #         df = pd.concat([df, new_feature(split)], ignore_index=True)
    #     split = None
    #     print(df)
    df = new_feature(df)
    # drop_bytes(df)

    # assert not df.isnull().values.any(axis=None)

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