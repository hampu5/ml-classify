import matplotlib
# matplotlib.use("pgf")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns

from compiledataset import load_dataset, get_datasets, compile_dataset, merge_data_features


datasets = {}


PATH_ORNL = "/home/hampus/miun/master_thesis/Datasets/ORNL/"
PATH_SURVIVAL = "/home/hampus/miun/master_thesis/Datasets/Survival/"
PATH_HISINGEN = "/home/hampus/miun/master_thesis/Datasets/Hisingen/"


dataset: pd.DataFrame = load_dataset(PATH_ORNL, "data.csv")
dataset["remarks"] = "No DLC available"
datasets["ROAD"] = dataset.to_dict("records")

dataset: pd.DataFrame = load_dataset(PATH_SURVIVAL, "data.csv")
dataset["remarks"] = "-"
datasets["Survival"] = dataset.to_dict("records")

# dataset: pd.DataFrame = load_dataset(PATH_HISINGEN, "data.csv")
# dataset["remarks"] = "-"
# datasets["Hisingen"] = dataset.to_dict("records")

dataset = None # Release memory, as it isn't used for now


df_attack, df_ambient = compile_dataset(datasets)
df_all = pd.concat([df_attack, df_ambient], ignore_index=True)

df_attack = None # Release memory
df_ambient = None # Release memory

df_all.drop(columns=["DLC", "t", "type"], inplace=True, errors="ignore")

X_sampled = df_all.drop(columns="Label")
y_sampled = df_all["Label"]

rus = RandomUnderSampler(random_state=0)
X_sampled, y_sampled = rus.fit_resample(X_sampled, y_sampled)

X_sampled, _, y_sampled, _ = train_test_split(X_sampled, y_sampled, train_size=0.001, random_state=2, shuffle=True, stratify=y_sampled)

df_all = pd.concat([X_sampled, y_sampled], axis=1)
print(df_all)

df_all.drop(columns="Label", inplace=True, errors="ignore")

# Compute the correlation matrix
corr = df_all.corr()

# # Drop first row and last column that don't provide information
# corr.drop(index=corr.index[0], inplace=True)
# corr.drop(columns=corr.columns[-1], inplace=True)

# Generate a mask for the upper triangle but not the diagonal
mask = np.triu(np.ones_like(corr, dtype=bool))
np.fill_diagonal(mask, False)

# Draw the correlation heatmap with the mask
def tostr(num):
    if isinstance(num, str): return num
    if num < 0: return str(num)[:5]
    return str(num)[:4]
def remove_nocorr(corr):
    annot = corr.copy()
    annot.where(np.abs(annot) > 0.2, " ", inplace=True)
    annot = annot.applymap(tostr)
    return annot
annots = remove_nocorr(corr)

# Create plot
sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0, annot=annots, annot_kws={"fontsize": 8}, fmt="s")

# df_fuzz = df_all[df_all["type"] == "fabr"]
# sns.pairplot(data=df_fuzz) #, hue="Label")

# sns.scatterplot(data=df_fuzz, x="dt", y="dt_ID") #, hue="Label")

# df_all.drop(columns=["ID", "dt", "dt_ID", "Label"], inplace=True, errors="ignore")
# sns.boxplot(data=df_all, x="dt_ID", y="type", orient="h")

plt.show()
# plt.savefig("Survival_correlation_heatmap.pgf")