import pandas as pd

PATH = "/home/hampus/miun/master_thesis/Datasets/road/signal_extractions/attacks/correlated_signal_attack_1_masquerade.csv"

def create_dt(df: pd.DataFrame):
    dt = df["t"].diff()
    dt.iloc[0] = dt.mean() # fill the missing first element
    df["dt"] = dt

def create_dt_ID(df: pd.DataFrame):
    df["dt_ID"] = df.groupby(by="ID")["t"].diff()

    meanall = df["dt_ID"].mean() # needed when an ID is used only once, hence no mean
    df["dt_ID"] = df.groupby("ID")["dt_ID"].apply(lambda x: x.fillna(x.mean() if len(x) > 1 else meanall))

def load_signal_road():
    df = pd.read_csv(PATH)
    
    df.rename(columns={'Timestamp':'t', 'Time':'t'}, inplace=True, errors="ignore")

    # create_dt(df)
    # create_dt_ID(df)

    return df