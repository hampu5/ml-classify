import numpy as np
import pandas as pd


path = "/home/hampus/miun/master_thesis/Datasets/Hisingen/"

df1 = pd.read_csv(path + "data.csv")

df_attack = pd.DataFrame()
for index, dataitem in df1.iterrows():
    name = dataitem["name"]
    filename = dataitem["filename"]
    
    df = pd.read_csv(path + filename)
    print(name)
    print(df)
    print(np.amax(df["d0"]))