import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use("pgf")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, pred, title, cbar=True):
    cm_norm = confusion_matrix(y_test, pred, normalize="true")
    cm = confusion_matrix(y_test, pred)
    annots = pd.DataFrame(cm_norm).applymap(lambda x: str(round(x, 3)))
    annots += pd.DataFrame(cm).applymap(lambda x: f"\n({str(x)})")
    plt.figure(dpi=100)
    sns.heatmap(cm_norm, annot=annots, fmt="s", square=True, vmin=0, vmax=1, xticklabels=["normal", "attack"], yticklabels=["normal", "attack"], cbar=cbar)
    # plt.title(title)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame):
    # Compute the correlation matrix
    corr = df.corr()

    # Drop first row and last column that don't provide information
    corr.drop(index=corr.index[0], inplace=True)
    corr.drop(columns=corr.columns[-1], inplace=True)

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
        annot.where(np.abs(annot) > 0.1, " ", inplace=True)
        annot = annot.applymap(tostr)
        return annot
    
    annots = remove_nocorr(corr)

    plt.figure(dpi=100)
    sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0, annot=annots, annot_kws={"fontsize": 5}, fmt="s", square=True)
    plt.title("ROAD Dataset Correlation Matrix (Pearson)")
    plt.show()

 
def plot_pairplot(df: pd.DataFrame):
   sns.pairplot(data=df.sample(500, random_state=0), hue="Label")
   plt.show()
