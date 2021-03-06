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
def load_dataset(path, filename, name=None) -> pd.DataFrame:
    data = pd.read_csv(f"{path}/{filename}")    
    if name != None: data = data.loc[data["name"] == name]
    assert len(data) > 0
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


def binary_ID(df: pd.DataFrame):
    df["bin_ID"] = df["ID"].map(format_binary)

    assert no_nan_or_inf(df["bin_ID"])

# Various functions to create new features

def create_dc(df: pd.DataFrame):
    df["dc"] = df.groupby("ID")["data"].shift().ne(df['data']).astype(int)

    assert no_nan_or_inf(df["dc"])

def smc(s1, s2): # Simple Matching Coefficient
    return sum([1 for c1, c2 in zip(s1, s2) if c1 == c2]) / 64

def fuzzify(s):
    s_temp = [0] * len(s) # Always 64 in this study
    for i, b1 in enumerate(s):
        if b1 == "1":
            if i - 2 >= 0: s_temp[i-2] += 1
            if i - 1 >= 0: s_temp[i-1] += 2
            s_temp[i] += 4
            if i + 1 < len(s): s_temp[i+1] += 2
            if i + 2 < len(s): s_temp[i+2] += 1
    return s_temp

# 00110000111 11
# 13663113791097 <- FMC

# 0 1 2 3 4 5 6 7  8  9  10 11 12 13 <- Ascending integers
# 1 3 6 6 3 1 1 3  7  9  1  0  9  7  <- FMC
# 1 4 8 9 7 6 7 10 15 18 11 11 21 20 <- FMC Ascending

def fmc(s1, s2): # "Fuzzy" Matching Coefficient
    s1_fuzz = fuzzify(s1)
    s2_fuzz = fuzzify(s2)
    return sum([(10 - abs(c1-c2)) for c1, c2 in zip(s1_fuzz, s2_fuzz)]) / (64*10)

def fmc_ascending(s1, s2): # "Fuzzy" Matching Coefficient Ascending version
    s1_fuzz = fuzzify(s1)
    s2_fuzz = fuzzify(s2)
    return sum([(10 + n - abs(c1-c2)) for n, (c1, c2) in enumerate(zip(s1_fuzz, s2_fuzz))]) / (64*10*1.45)


def check_unique(x):
    shifted = x.shift(1)
    temp = {}
    for (idx1, s1), (idx2, s2) in zip(x.items(), shifted.items()):
        if not (isinstance(s1, str) and isinstance(s2, str)):
            temp[idx2] = np.nan
            continue
        temp[idx2] = 1 - fmc(s1, s2) # fmc might be more general
    return pd.Series(temp)

def create_dcs(df: pd.DataFrame):
    dcs = check_unique(df["data"])
    df["dcs"] = dcs.fillna(dcs.mean())

    assert no_nan_or_inf(df["dcs"])

def create_dcs_ID(df: pd.DataFrame):
    df["dcs_ID"] = df.groupby("ID")["data"].apply(check_unique)
    # df["dcs_ID"] = dcs_ID.fillna(dcs_ID.mean())
    meanall = df["dcs_ID"].mean() # needed when a dcs is used only once, hence no mean
    df["dcs_ID"] = df.groupby("ID")["dcs_ID"].apply(lambda x: x.fillna(x.mean() if len(x) > 1 else meanall))

    assert no_nan_or_inf(df["dcs_ID"])


def create_ics(df: pd.DataFrame):
    ics = check_unique(df["bin_ID"])
    df["ics"] = ics.fillna(ics.mean())
    
    assert no_nan_or_inf(df["ics"])

def create_ics_data(df: pd.DataFrame):
    df["ics_data"] = df.groupby("data")["bin_ID"].apply(check_unique)
    meanall = df["ics_data"].mean() # needed when a dcs is used only once, hence no mean
    df["ics_data"] = df.groupby("data")["ics_data"].apply(lambda x: x.fillna(x.mean() if len(x) > 1 else meanall))

    assert no_nan_or_inf(df["ics_data"])


# Helper function to translate absolute time to relative time
def create_dt(df: pd.DataFrame):
    dt = df["t"].diff()
    dt.iloc[0] = dt.mean() # fill the missing first element
    df["dt"] = dt

    assert no_nan_or_inf(df["dt"])

# Calculate relative time between IDs
def create_dt_ID(df: pd.DataFrame):
    df["dt_ID"] = df.groupby(by="ID")["t"].diff()

    meanall = df["dt_ID"].mean() # needed when an ID is used only once, hence no mean
    df["dt_ID"] = df.groupby("ID")["dt_ID"].apply(lambda x: x.fillna(x.mean() if len(x) > 1 else meanall))

    assert df["dt_ID"].min() >= 0
    assert no_nan_or_inf(df["dt_ID"])

def create_dt_data_ID(df: pd.DataFrame):
    df["dt_data"] = df.groupby("ID").apply(
        lambda x: x.groupby("data", as_index=False)["t"].diff())
    
    meanall = df["dt_data"].mean() # needed when a Data is used only once, hence no mean
    df["dt_data"] = df.groupby("data")["dt_data"].apply(lambda x: x.fillna(x.mean() if len(x) > 1 else meanall))

    assert no_nan_or_inf(df["dt_data"])

def create_dt_data(df: pd.DataFrame):
    df["dt_data"] = df.groupby(by="data")["t"].diff()

    meanall = df["dt_data"].mean() # needed when a Data is used only once, hence no mean
    df["dt_data"] = df.groupby("data")["dt_data"].apply(lambda x: x.fillna(x.mean() if len(x) > 1 else meanall))

    assert no_nan_or_inf(df["dt_data"])

def create_dt_data_bytewise(df: pd.DataFrame):
    for i in range(8):
        df[f"dt_d{i}"] = df.groupby(by=f"d{i}")["t"].diff()

        meanall = df[f"dt_d{i}"].mean() # needed when a Data is used only once, hence no mean
        df[f"dt_d{i}"] = df.groupby(f"d{i}")[f"dt_d{i}"].apply(lambda x: x.fillna(x.mean() if len(x) > 1 else meanall))

        assert no_nan_or_inf(df[f"dt_d{i}"])

from IPython.display import display
def count_ones(df: pd.DataFrame):
    # display(df["data"])
    df["ones"] = df["data"].apply(lambda x: x.count("1"))

    assert not df["ones"].isnull().values.any(axis=None)

def create_dt_ones(df: pd.DataFrame):
    count_ones(df)

    df["dt_ones"] = df.groupby(by="ones")["t"].diff()

    df["dt_ones"].fillna(df["dt_ones"].mean(), inplace=True)

    df.drop(columns="ones", inplace=True)

    assert no_nan_or_inf(df["dt_ones"])

def create_dt_runs(df: pd.DataFrame):
    df["one_runs"] = df["data"].apply(lambda x: len(list(filter(None, re.split("0+", x)))))
    # df["one_runs"] += df["data"].apply(lambda x: len(list(filter(None, re.split("1+", x)))))

    df["dt_runs"] = df.groupby(by="one_runs")["t"].diff()

    # df["dt_runs"].fillna(df["dt_runs"].mean(), inplace=True)
    meanall = df["dt_runs"].mean() # needed when a run is used only once, hence no mean
    df["dt_runs"] = df.groupby("one_runs")["dt_runs"].apply(lambda x: x.fillna(x.mean() if len(x) > 1 else meanall))

    df.drop(columns="one_runs", inplace=True)

    assert no_nan_or_inf(df["dt_runs"])

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



def get_labels_by_class(class_, window_size): 
    # return [c for c in df.columns if c.startswith(class_ + "_")]    
    return [ f"{class_}_{i}" for i in range(window_size) ]

#Features related to frequency using window
def extract_window(df: pd.DataFrame, window_size):

    cols = df.columns
    dfw = pd.DataFrame()
    id_column_names = [] #added
    for i in range(window_size):
        tmp = df.shift(-i)
        column_names = {c: f"{c}_{i}" for c in cols}
        id_column_names.append(column_names['ID']) #added
        tmp = tmp.rename(columns=column_names)
        if dfw.shape[0] == 0:
            dfw = tmp
        else:
            dfw = dfw.join(tmp)
    
    print('ID NAMES: %s' % id_column_names) #added

    
    ## Calculate 
    from collections import Counter #added
    dfw["win_most_freq_id"] = dfw.apply(lambda x: list([x[name] for name in id_column_names]), axis=1) #added
    dfw["win_most_freq_id"] = dfw["win_most_freq_id"].apply(lambda x: max(Counter(x).values())) #added


    # drop the last ones
    len_ = dfw.shape[0]
    dfw.drop(range(len_ - window_size, len_), inplace=True)

    # compute the labels and remove the individual labels    
    lbs = get_labels_by_class("Label", window_size)    
    dfw["Label"] = dfw[lbs].max(axis=1)
    dfw.drop(columns=lbs, inplace=True)
    
    # compute win_dur and win_mean_dt:
    dts = get_labels_by_class("dt", window_size)    
    dfw["win_dur"] = dfw[dts].sum(axis=1)
    dfw["win_mean_dt"] = dfw[dts].mean(axis=1)
    
    # drop the absolut times, they are useless now
    ts = get_labels_by_class("t", window_size)    
    dfw.drop(columns=ts, inplace=True)
    
    
    # Y = dfw["Label"]
    # X = dfw.drop("Label", axis = 1)
    return dfw



def read_file(filename):
    df = pd.read_csv(filename)
    
    df.rename(columns={'Timestamp':'t'}, inplace=True, errors="ignore")

    binary_payload(df)
    binary_ID(df)
    # dec_payload(df)
    # drop_bytes(df)

    create_dcs(df)
    create_dcs_ID(df)

    create_ics(df)
    create_ics_data(df)

    create_dt(df)
    create_dt_ID(df)
    create_dt_data(df)
    # create_dt_data_bytewise(df)
    # create_dt_ones(df)
    # create_dt_runs(df)
    # create_dt_ID_data(df)
    # create_dc(df)

    # df.drop(columns=["data", "data_dec", "DLC"], inplace=True, errors="ignore")
    # df.drop(columns=["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"], inplace=True, errors="ignore")

    # df = extract_window(df, 10)
    
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
            
            if atype == "masq": continue
            # if name != "Vehicle_F-Model_2-Fabrication_attack-Sample_1": continue
            # if c != "Sonata": continue

            df = read_file(filename)
            df["name"] = name
            df["class"] = c
            df["dataset"] = dname
            df["type"] = atype
            # df["type"] = "none"
            # df.loc[df["Label"] == 1, "type"] = atype
            # print(df["type"])
            df_all = pd.concat([df_all, df], ignore_index=True)
    
    df_all = df_all[[c for c in df_all if c not in ["dataset", "type", "Label"]] + ["dataset", "type", "Label"]]

    # assert not df_all.isnull().values.any(axis=None)

    return df_all