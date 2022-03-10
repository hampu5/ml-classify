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


# Helpers

def count_bit(val, bit: str):
    if isinstance(val, str):
        return val.count(bit)
    return bin(val).count(bit)

def count_bit_bins(val, bit: str):
    bit = "1" if bit == "0" else "0"
    if isinstance(val, str):
        return len(list(filter(None, re.split(f"{bit}+", val))))
    return len(list(filter(None, re.split(f"{bit}+", bin(val)))))

def format_binary(val):
    return f"{val:08b}"

def no_nan_or_inf(s: pd.Series):
    s.replace([np.inf, -np.inf], np.nan, inplace=True)
    return not s.isnull().values.any(axis=None)


# Loading different datasets
def load_dataset(path, filename) -> pd.DataFrame:
    data = pd.read_csv(f"{path}/{filename}")
    data["filename"] = data["filename"].apply(lambda filename: f"{path}/{filename}")
    data = data[["name", "type", "filename", "has_attack"]]
    return data

# Get all smaller separate datasets within a dataset
def get_datasets(dataset):
    datasets = {}
    for index, dataitem in dataset.iterrows():
        name = dataitem["name"]
        filename = dataitem["filename"]
        df = pd.read_csv(filename)
        datasets[name] = df
    return datasets


# various functions to get the data into the format we want

# helper function to translate absolute time to relative time
def calc_relative_time(df: pd.DataFrame):
    dt = df["t"].diff()
    dt.iloc[0] = dt.mean() # fill the missing first element
    df["dt"] = dt

    assert not df["dt"].isnull().values.any(axis=None)

# Calculate relative time between IDs
def calc_relative_time_per_id(df: pd.DataFrame):
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

    assert not df["dt_ID"].isnull().values.any(axis=None)
    # assert df["dt_ID"].isnull().sum() == 0

def read_file(filename):
    df = pd.read_csv(filename)
    
    if "Timestamp" in df.columns:
        df.rename(columns={'Timestamp':'t'}, inplace=True)
    
    calc_relative_time(df)
    calc_relative_time_per_id(df)
    
    return df

def merge_data_features(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    
    number_of_bytes = 8
    for index, col in enumerate(df_data):
        df_data[col] = df_data[col].apply(lambda val: val * pow(256, number_of_bytes - index)).astype(np.float64)
        # df_data[col] = df_data[col].apply(lambda val: hex(val))

    df["data"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    # df["data"] = df["data"].apply(lambda val: bin)
    # df["data"] = (df["data"] - df["data"].min()) / (df["data"].max() - df["data"].min())

    return df


def count_ones(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    # ones = pd.Series
    df["ones"] = 0
    for col in df_data:
        temp = df_data[col].apply(count_bit, args="1")
        df["ones"] += temp
    
    df_data = None

    assert not df["ones"].isnull().values.any(axis=None)

    df["ones_ID"] = df["ID"].apply(count_bit, args="1")

    assert not df["ones_ID"].isnull().values.any(axis=None)

def get_binary_payload(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    drop_bytes(df)

    df_data = df_data.applymap(format_binary)
    
    df["bin_data"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    
    df_data = None
    
    assert not df["bin_data"].isnull().values.any(axis=None)

def count_runs(df: pd.DataFrame):
    get_binary_payload(df)

    df["one_runs"] = df["bin_data"].apply(count_bit_bins, args="1")
    df["zero_runs"] = df["bin_data"].apply(count_bit_bins, args="0")
    df.drop(columns="bin_data", inplace=True)

    assert no_nan_or_inf(df["one_runs"])
    assert no_nan_or_inf(df["zero_runs"])

def count_runs_diff(df: pd.DataFrame):
    get_binary_payload(df)

    one_runs = df["bin_data"].apply(count_bit_bins, args="1")
    zero_runs = df["bin_data"].apply(count_bit_bins, args="0")
    df.drop(columns="bin_data", inplace=True)

    df["run_diff"] = one_runs / (zero_runs + 1) # +1 to not divide by 0

    assert no_nan_or_inf(df["run_diff"])

    one_runs = df["ID"].apply(count_bit_bins, args="1")
    zero_runs = df["ID"].apply(count_bit_bins, args="0")

    df["run_diff_ID"] = one_runs / (zero_runs + 1)

    assert no_nan_or_inf(df["run_diff_ID"])


def count_ones_weighted(df: pd.DataFrame):
    get_binary_payload(df)
    
    zeros = df["bin_data"].apply(count_bit, args="0")
    df["ones_w"] = df["bin_data"].apply(count_bit, args="1")
    df["ones_w"] *= df["bin_data"].apply(count_bit_bins, args="0")
    df.drop(columns="bin_data", inplace=True)
    
    zeros.replace(0, 1, inplace=True)
    df["ones_w"] /= zeros
    zeros = None

    df["ones_w"].replace([np.inf, -np.inf], np.nan, inplace=True)

    assert not df["ones_w"].isnull().values.any(axis=None)

def count_ones_weighted2(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    drop_bytes(df)

    df["ones_w"] = 0
    zeros = df["ones_w"].copy()
    df["weights"] = 0
    for col in df_data:
        print(col)
        zeros += df_data[col].apply(count_bit, args="0")
        df["ones_w"] += df_data[col].apply(count_bit, args="1")
        df["weights"] += df_data[col].apply(count_bit_bins, args="1")
    
    df["ones_w"] *= df["weights"]
    df.drop(columns="weights", inplace=True)

    zeros.replace(0, 1, inplace=True)
    df["ones_w"] /= zeros
    zeros = None

    df["ones_w"].replace([np.inf, -np.inf], np.nan, inplace=True)

    assert not df["ones_w"].isnull().values.any(axis=None)

def drop_bytes(df: pd.DataFrame):
    df.drop(columns=["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"], inplace=True, errors="ignore")

def new_feature(df: pd.DataFrame):
    # count_ones_weighted2(df)
    # count_ones_weighted(df)
    count_ones(df)
    # count_runs(df)
    count_runs_diff(df)
    # df = merge_data_features(df)

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
            # name = dataitem["name"]
            atype = dataitem["type"]
            filename = dataitem["filename"]
            has_attack = bool(dataitem["has_attack"])
            # remarks = dataitem["remarks"] or ""
            df = read_file(filename)
            # df["name"] = name
            df["dataset"] = dname
            df["type"] = "none"
            df.loc[df["Label"] == 1, "type"] = atype
            # print(df["type"])
            if has_attack:
                df_attack = pd.concat([df_attack, df], ignore_index=True)
            else:
                df_ambient = pd.concat([df_ambient, df], ignore_index=True)
    
    df_all = pd.concat([df_attack, df_ambient], ignore_index=True)
    df_all = feature_creation(df_all)
    df_all = df_all[[c for c in df_all if c not in ["dataset", "type", "Label"]] + ["dataset", "type", "Label"]]

    # assert not df_all.isnull().values.any(axis=None)

    return df_all