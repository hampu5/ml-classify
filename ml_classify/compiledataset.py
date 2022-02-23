import numpy as np
import pandas as pd
import re


# Loading different datasets
def load_dataset(path, filename) -> pd.DataFrame:
    data = pd.read_csv(f"{path}/{filename}")
    data["filename"] = data["filename"].apply(lambda x: path + x)
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
def calc_relative_time(AT):
    dt = AT.diff()
    dt[0] = dt.mean() # fill the missing first element
    return dt

# Calculate relative time between IDs
def calc_relative_time_per_id(df):
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

    assert df.dt_ID.isnull().sum() == 0

def read_file(filename):
    df = pd.read_csv(filename)
    
    if "Timestamp" in df.columns:
        df.rename(columns={'Timestamp':'t'}, inplace=True)
    
    df["dt"] = calc_relative_time(df["t"])
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

def get_binary_payload(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    df_data = df_data.apply(lambda col: col.apply(lambda val: f"{val:08b}"))

    df["bin_data"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    print(df)
    
    return df

def count_ones(df: pd.DataFrame):
    df_data = df[["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]]
    ones = pd.Series
    df["ones"] = 0
    for col in df_data:
        temp = df_data[col].apply(lambda val: bin(val).count("1"))
        df["ones"] += temp
    
    return df

def count_ones_weighted(df: pd.DataFrame):
    df = get_binary_payload(df)
    ones = df["bin_data"].apply(lambda val: val.count("1"))
    weight_list = df["bin_data"].apply(lambda val: len(list(filter(None, re.split("1+", val)))) )
    weights = (64 - ones) / weight_list
    
    print(weights)

    df["ones_w"] = ones / weights

    df.drop(columns="bin_data", inplace=True)

    return df

def drop_bytes(df: pd.DataFrame):
    df.drop(columns=["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"], inplace=True)

pd.options.mode.chained_assignment = None # Chained assignment warning
def compile_dataset(datasets):
    df_attack = pd.DataFrame()
    df_ambient = pd.DataFrame()

    for dname, dataset in datasets.items():
        for dataitem in dataset:
            name = dataitem["name"]
            atype = dataitem["type"]
            filename = dataitem["filename"]
            has_attack = bool(dataitem["has_attack"])
            remarks = dataitem["remarks"] or ""
            df = read_file(filename)
            # df["name"] = name
            df["type"] = "none"
            # df.loc["type", df["Label"] == 1] = atype
            df["type"][df["Label"] == 1] = atype

            if has_attack:
                df_attack = pd.concat([df_attack, df], ignore_index=True)
            else:
                df_ambient = pd.concat([df_ambient, df], ignore_index=True)

    df_attack = count_ones_weighted(df_attack)
    df_ambient = count_ones_weighted(df_ambient)
    # df_attack = count_ones(df_attack)
    # df_ambient = count_ones(df_ambient)
    # df_attack = merge_data_features(df_attack)
    # df_ambient = merge_data_features(df_ambient)

    drop_bytes(df_attack)
    drop_bytes(df_ambient)

    df_attack = df_attack[[c for c in df_attack if c not in ["Label"]] + ["Label"]]
    df_ambient = df_ambient[[c for c in df_ambient if c not in ["Label"]] + ["Label"]]
    return df_attack, df_ambient