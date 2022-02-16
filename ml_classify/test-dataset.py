import matplotlib
matplotlib.use("pgf")
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
import seaborn as sns

# Maybe it is possible to account for obfuscation by using some kind of shape transformation instead of relying on values
# in the payload.

from compiledataset import load_dataset, get_datasets, compile_dataset

datasets = {}


PATH_ORNL = "/home/hampus/miun/master_thesis/Datasets/ORNL/"
PATH_SURVIVAL = "/home/hampus/miun/master_thesis/Datasets/Survival/"
PATH_HISINGEN = "/home/hampus/miun/master_thesis/Datasets/Hisingen/"


dataset: pd.DataFrame = load_dataset(PATH_ORNL, "data.csv")
dataset["remarks"] = "No DLC available"
datasets["ROAD"] = dataset.to_dict("records")

# dataset: pd.DataFrame = load_dataset(PATH_SURVIVAL, "data.csv")
# dataset["remarks"] = "-"
# datasets["Survival"] = dataset.to_dict("records")

# dataset: pd.DataFrame = load_dataset(PATH_HISINGEN, "data.csv")
# dataset["remarks"] = "-"
# datasets["Hisingen"] = dataset.to_dict("records")

dataset = None # Release memory, as it isn't used for now


df_attack, df_ambient = compile_dataset(datasets)
df_all = pd.concat([df_attack, df_ambient], ignore_index=True)
df_attack = None # Release memory
df_ambient = None # Release memory


df_all.drop(columns=["DLC", "t", "dt", "dt_ID"], inplace=True, errors="ignore")

print(df_all)

X = df_all.drop(columns="Label")
y = df_all["Label"]

df_all = None # Release memory

# Split dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, shuffle=True, stratify=y)
print("Test and training data created!")
print(f"Train: {np.bincount(y_train)} Test: {np.bincount(y_test)}")

X = None # Release memory
y = None # Release memory


# Classification with Random Forest
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


scores = cross_val_score(clf, X_train, y_train, scoring='f1', cv=10, n_jobs=-1)
print("Training F1: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

pred = clf.predict(X_test)
print("Test data has been Classified!")

f1_scores = f1_score(y_test, pred, average='weighted')
print("Testing F1:  %0.4f(+/- %0.4f)" % (f1_scores.mean(), f1_scores.std()))

shap.initjs()
explainer = shap.TreeExplainer(clf)
print("Explainer created!")
shap_values = explainer.shap_values(X_train)
print("Shap values created!")
dump(shap_values, "RF_Survival_Shap.joblib")

# shap_values = load("RF_ROAD_Shap.joblib")

# Make sure that the ingested SHAP model (a TreeEnsemble object) makes the
# same predictions as the original model
assert np.abs(explainer.model.predict(X_test) - clf.predict(X_test)).max() < 1e-4

# make sure the SHAP values sum up to the model output (this is the local accuracy property)
assert np.abs(explainer.expected_value + explainer.shap_values(X_test).sum(1) - clf.predict(X_test)).max() < 1e-4

shap.summary_plot(shap_values[1], X_train)