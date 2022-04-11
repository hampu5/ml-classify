from math import ceil
import string
import numpy as np
import pandas as pd
import re


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
    
    df["data_hex"] = df_data["d0"] + df_data["d1"] + df_data["d2"] + df_data["d3"] + df_data["d4"] + df_data["d5"] + df_data["d6"] + df_data["d7"]
    
    df_data = None
    
    assert not df["data_hex"].isnull().values.any(axis=None)


def read_file(filename):
    df = pd.read_csv(filename)
    
    df.rename(columns={'Timestamp':'t'}, inplace=True, errors="ignore")
    
    binary_payload(df)
    hex_payload(df)
    # dec_payload(df)
    # drop_bytes(df)
    
    return df



def compile_dataset(datasets: dict):
    df_all = pd.DataFrame()

    for dname, dataset in datasets.items():
        for dataitem in dataset:
            name = dataitem["name"]
            c = dataitem["class"]
            atype = dataitem["type"]
            filename = dataitem["filename"]
            has_attack = bool(dataitem["has_attack"])
            if name != "FreeDrivingData_20180323_SONATA":
                continue
            # remarks = dataitem["remarks"] or ""
            df = read_file(filename)
            df["name"] = name
            df["class"] = c
            df["dataset"] = dname
            df["type"] = atype
            df_all = pd.concat([df_all, df], ignore_index=True)
    
    df_all = df_all[[c for c in df_all if c not in ["dataset", "type", "Label"]] + ["dataset", "type", "Label"]]

    # assert not df_all.isnull().values.any(axis=None)

    return df_all