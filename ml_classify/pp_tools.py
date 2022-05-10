import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, RobustScaler

def scale_dataset(df: pd.DataFrame, only_normal=False):

    numeric_columns = [col for col in df if is_numeric_dtype(df[col].dtype)]

    # feature_columns= list(set(df.columns.to_list()).difference(["name", "class", "dataset", "type", "Label"]))

    scaler = None
    if only_normal:
        scaler = StandardScaler().fit(df.loc[df["Label"] == 0, numeric_columns])
    else:
        scaler = StandardScaler().fit(df.loc[:, numeric_columns])

    df.loc[:, numeric_columns] = scaler.transform(df.loc[:, numeric_columns])

    # for col in feature_columns:
        
    #     if only_normal:
    #         scaler = StandardScaler().fit(df.loc[df["Label"] == 0, col])
    #     else:
    #         scaler = StandardScaler().fit(df.loc[:, col])
        
    #     df.loc[:, col] = scaler.transform(df.loc[:, col])