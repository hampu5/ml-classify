import numpy as np
import pandas as pd


# Loading different datasets
def load_dataset(path, filename) -> pd.DataFrame:
    data = pd.read_csv(f"{path}/{filename}")
    data["filename"] = data["filename"].apply(lambda x: path + x)
    data = data[["name", "filename", "has_attack"]]
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

def compile_dataset(datasets):
    df_attack = pd.DataFrame()
    df_ambient = pd.DataFrame()

    for dname, dataset in datasets.items():
        for dataitem in dataset:
            name = dataitem["name"]
            filename = dataitem["filename"]
            has_attack = bool(dataitem["has_attack"])
            remarks = dataitem["remarks"] or ""
            
            if has_attack:
                df = read_file(filename)
                df_attack = pd.concat([df_attack, df], ignore_index=True)
            else:
                df = read_file(filename)
                df_ambient = pd.concat([df_ambient, df], ignore_index=True)
    df_attack = df_attack[[c for c in df if c not in ["Label"]] + ["Label"]]
    df_ambient = df_ambient[[c for c in df if c not in ["Label"]] + ["Label"]]
    return df_attack, df_ambient