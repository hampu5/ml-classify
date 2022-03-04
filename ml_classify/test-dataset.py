from timeit import timeit
import matplotlib
# matplotlib.use("pgf")
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from joblib import dump, load
from imblearn.under_sampling import RandomUnderSampler
import shap
import seaborn as sns

# Maybe it is possible to account for obfuscation by using some kind of shape transformation instead of relying on values
# in the payload.

from compiledataset import load_dataset, get_datasets, merge_data_features, compile_dataset

datasets = {}

PATH_ORNL = "/home/hampus/miun/master_thesis/Datasets/ORNL"
PATH_SURVIVAL = "/home/hampus/miun/master_thesis/Datasets/Survival"
PATH_HISINGEN = "/home/hampus/miun/master_thesis/Datasets/Hisingen"


dataset: pd.DataFrame = load_dataset(PATH_ORNL, "data_a.csv")
dataset["remarks"] = "No DLC available"
datasets["ROAD"] = dataset.to_dict("records")

# dataset: pd.DataFrame = load_dataset(PATH_SURVIVAL, "data.csv")
# dataset["remarks"] = "-"
# datasets["Survival"] = dataset.to_dict("records")

# dataset: pd.DataFrame = load_dataset(PATH_HISINGEN, "data.csv")
# dataset["remarks"] = "-"
# datasets["Hisingen"] = dataset.to_dict("records")

dataset = None # Release memory, as it isn't used for now

df_all = compile_dataset(datasets)
# df_attack, df_ambient = compile_dataset(datasets)
# df_all = pd.concat([df_attack, df_ambient], ignore_index=True)
# attack_mean = df_attack["dt"].min()
# ambient_mean = df_ambient["dt"].min()
# all_mean = df_all["dt"].min()
# print(f"attacks dt min: {attack_mean}")
# print(f"ambient dt min: {ambient_mean}")
# print(f"all dt min: {all_mean}")

# df_attack = None # Release memory
# df_ambient = None # Release memory


# grouped = df.groupby(df.color)
# df_new = grouped.get_group("E")


df_all.drop(columns=["DLC", "t", "data", "ID"], inplace=True, errors="ignore")



X_sampled = df_all.drop(columns="type")
y_sampled = df_all["type"]

df_all = None # Release memory



# # Use under-sampling on the majority Label (0, no attack)
# rus = RandomUnderSampler(random_state=0)
# X_sampled, y_sampled = rus.fit_resample(X_sampled, y_sampled)

# d_temp: pd.DataFrame = pd.concat([X_sampled, y_sampled], axis="columns")
# X_sampled = d_temp.drop(columns="type")
# y_sampled = d_temp["type"]
# d_temp = None

# Split dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=0, shuffle=True, stratify=y_sampled)

X_sampled = None # Release memory
y_sampled = None # Release memory

print("Test and training data created!")
# print(f"Types (after stratification)\n\tTrain: {np.bincount(y_train)} Test: {np.bincount(y_test)}")

# X_train.drop(columns="type", inplace=True, errors="ignore")
# X_test.drop(columns="type", inplace=True, errors="ignore")


# print(f"Types (after under-sampling)\n\tTrain: {np.bincount(y_train)} Test: {np.bincount(y_test)}")

d_temp: pd.DataFrame = pd.concat([X_train, y_train], axis="columns")
# d_temp.drop(columns="type", inplace=True, errors="ignore")
X_train = d_temp.drop(columns="Label")
y_train = d_temp["Label"]
d_temp: pd.DataFrame = pd.concat([X_test, y_test], axis="columns")
# d_temp.drop(columns="type", inplace=True, errors="ignore")
X_test = d_temp.drop(columns="Label")
y_test = d_temp["Label"]
d_temp = None

# Use under-sampling on the majority Label (0, no attack)
rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_resample(X_train, y_train)

print(f"Labels\n\tTrain: {np.bincount(y_train)} Test: {np.bincount(y_test)}")


# # Chi Squared test

# chi_scores = chi2(X_train, y_train)
# p_values = pd.Series(chi_scores[1], index=X_train.columns)
# p_values.sort_values(ascending=False, inplace=True)
# p_values.plot.bar()
# plt.show()
# exit()

# Distribution of Labels to counted ones
df_all = pd.concat([X_train, y_train], axis="columns")
size = len(df_all.loc[df_all["Label"] == 1].index)
ones_prob = [[], [], []]
# zeros_prob = []
for i in range(65):
    # zeros_prob.append(len(df_all.loc[(df_all["Label"] == 0) & (df_all["ones"] == i)].index) / size)

    prob_flood = len(df_all.loc[(df_all["Label"] == 1) & (df_all["ones"] == i) & (df_all["type"] == "flood")].index) / size
    prob_fuzz = len(df_all.loc[(df_all["Label"] == 1) & (df_all["ones"] == i) & (df_all["type"] == "fuzz")].index) / size
    prob_fabr = len(df_all.loc[(df_all["Label"] == 1) & (df_all["ones"] == i) & (df_all["type"] == "fabr")].index) / size
    ones_prob[0].append(prob_flood)
    ones_prob[1].append(prob_fuzz)
    ones_prob[2].append(prob_fabr)


# zeros_prob = pd.Series(zeros_prob)
ones_prob = pd.Series(ones_prob)
# sns.lineplot(data=ones_prob)

plt.bar(x=range(0, 65), height=ones_prob[0])
# plt.bar(x=range(0, 65), height=ones_prob[1])
plt.bar(x=range(0, 65), height=ones_prob[2])
plt.legend(labels=["Masquerading", "Fuzzing", "Fabrication"])
# plt.bar(x=range(0, 65), height=zeros_prob)
plt.xlabel("Number of ones counted")
plt.ylabel("% of no attack observations (Label = 1)")
plt.show()

# df_attacktype = df_all.loc[(df_all["Label"] == 1) & (df_all["type"] == "fabr")]
# print(df_attacktype)
# print(f"bins: {np.bincount(df_attacktype.d0)}")

exit()


# Classification with Random Forest
clf = RandomForestClassifier(n_estimators=20, random_state=0, max_leaf_nodes=300).fit(X_train, y_train)
print("Random Forest model fitted!")
avg_depth = 0
avg_leaves = 0
for clf_est in clf.estimators_:
    depth = clf_est.get_depth()
    leaves = clf_est.get_n_leaves()
    # print(f"{depth}       {leaves}")
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

kappa_scores = cohen_kappa_score(y_test, pred)
print("Kappa score:  %0.4f(+/- %0.4f)" % (kappa_scores.mean(), kappa_scores.std()))

cm = confusion_matrix(y_test, pred)
print(cm)

# shap.initjs()
explainer = shap.TreeExplainer(clf)
print("Explainer created!")
shap_values = explainer(X_test.sample(1000, random_state=0))
print("Shap values created!")
# est_time = timeit(lambda: explainer(X_test[:1]), number=1)
shap_values = shap.Explanation(shap_values[:, :, 1], feature_names=X_test.columns)

# print(est_time)
# print(X_test[:1])
# print(shap_values)

# dump(shap_values, "RF_Survival_Shap.joblib")

# shap_values = load("RF_ROAD_Shap.joblib")

# # Make sure that the ingested SHAP model (a TreeEnsemble object) makes the
# # same predictions as the original model
# assert np.abs(explainer.model.predict(X_test) - clf.predict(X_test)).max() < 1e-4

# # make sure the SHAP values sum up to the model output (this is the local accuracy property)
# assert np.abs(explainer.expected_value + explainer.shap_values(X_test).sum(1) - clf.predict(X_test)).max() < 1e-4


# shap.waterfall_plot(shap.Explanation(values=shap_values[int("which_class")][row], 
#                                          base_values=explainer.expected_value[int(which_class)], 
#                                          data=X_test.iloc[row],  # added this line
#                                          feature_names=X_test.columns.tolist()))
# shap.force_plot(explainer.expected_value[1], shap_values[1], features=X_test[:1], feature_names=X_test.columns)

# shap.plots.scatter(shap_values[:,"ones_w"])
# shap.summary_plot(shap_values[1], X_test.columns)
shap.plots.beeswarm(shap_values)
# plt.show()