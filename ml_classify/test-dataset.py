from tokenize import Ignore
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from joblib import dump, load
import shap
# import seaborn as sns


datasets = {}


# ORNL DATA
PATH_ORNL = "/home/hampus/miun/master_thesis/Datasets/ORNL/"
PATH_SURVIVAL = "/home/hampus/miun/master_thesis/Datasets/Survival/"
PATH_HISINGEN = "/home/hampus/miun/master_thesis/Datasets/Hisingen/"

def load_dataset(path, filename, has_attacks):
    data = pd.read_csv(f"{path}/{filename}")
    data["filename"] = data["filename"].apply(lambda x: path + x)
    data["has_attacks"] = has_attacks
    data = data[["name", "filename", "has_attacks"]]
    return data


ambient = load_dataset(PATH_ORNL, "ambient.csv", False)
attack = load_dataset(PATH_ORNL, "attack.csv", True)

df1 = pd.concat([ambient, attack])
df1["remarks"] = "No DLC available"
datasets["ROAD"] = df1.to_dict("records")

# Release memory
ambient = None
attack = None

# df1 = load_dataset(PATH_SURVIVAL, "data.csv", True)
# df1["remarks"] = "-"
# datasets["Survival"] = df1.to_dict("records")

# df1 = load_dataset(PATH_HISINGEN, "data.csv", True)
# df1["remarks"] = "-"
# datasets["Hisingen"] = df1.to_dict("records")


# Release memory
df1 = None


# various functions to get the data into the format we want
def calc_relative_time(AT):    
    # helper function to translate absolute time to relative time
    dt = AT.diff()
    dt[0] = dt.mean() # fill the missing first element
    return dt

def calc_relative_time_per_id(df):
    df["dt_ID"] = df.groupby(by="ID")["t"].diff()

    nans_idx = df["dt_ID"].index[df["dt_ID"].apply(np.isnan)]
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
            has_attacks = bool(dataitem["has_attacks"])
            remarks = dataitem["remarks"] or ""
            
            if has_attacks:
                df = read_file(filename)
                df_attack = pd.concat([df_attack, df], ignore_index=True)
            # else:
            #     df = read_file(filename)
            #     df_ambient = pd.concat([df_ambient, df], ignore_index=True)
    return df_attack, df_ambient


df_attack, df_ambient = compile_dataset(datasets)
df_ambient = None # Release memory, as it isn't used for now

df_attack.drop("DLC", axis=1, inplace=True, errors="ignore")
# df_attack = df_attack[:1000]

print(df_attack)
# print(df_ambient)

attack_Y = df_attack["Label"]
attack_X = df_attack.drop("Label", axis=1)

# for col in df_attack:
#     print(df_attack[col].max())

X_train, X_test, y_train, y_test = train_test_split(attack_X, attack_Y, test_size=0.3, random_state=2, shuffle=True, stratify=attack_Y)
print("Test and training data created!")
print(f"Train: {np.bincount(y_train)} Test: {np.bincount(y_test)}")

# Free memory
# attack_X = None
attack_Y = None


#Classification with Random Forest
clf = RandomForestClassifier(n_estimators=20, random_state=0, max_depth=20, max_leaf_nodes=50).fit(X_train, y_train)
print("Random Forest model fitted!")
avg_depth = 0
avg_leaves = 0
for clf_est in clf.estimators_:
    depth = clf_est.get_depth()
    leaves = clf_est.get_n_leaves()
    print(f"{depth}       {leaves}")
    avg_depth += depth
    avg_leaves += leaves
avg_depth /= len(clf.estimators_)
avg_leaves /= len(clf.estimators_)
print(f"Average depth of trees: {avg_depth}     Average # of leaves: {avg_leaves}")

# dump(clf, "RF_ROAD.joblib")
# print("Model saved!")

# clf = load("RF_Survival.joblib")


# scores = cross_val_score(clf, X_train, y_train, scoring='f1', cv=10, n_jobs=-1)
# print("Training F1: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

# pred = clf.predict(X_test)
# print("Test data has been Classified!")

# f1_scores = f1_score(y_test, pred, average='weighted')
# print("Testing F1:  %0.4f(+/- %0.4f)" % (f1_scores.mean(), f1_scores.std()))


explainer = shap.Explainer(clf)
print("Explainer created!")
shap_values = explainer(attack_X)
print("Shap values created!")
dump(shap_values, "RF_ROAD_Shap.joblib")
shap.plots.beeswarm(shap_values)