import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
import shap

# Only works for binary classification
# 'X' and 'y' has the same size as 'shap_all' (the shap values for X and y)
# 'feature' is the feature to be plotted, trim is the x-axis range to plot, e.g., (0, 0.03)
# 'y_squish' is to shrink the y-axis if points cannot be placed within the plot
# 'colorbar' adds a color bar, and 'y_size' is used to compensate for size changes when adding color bar.

def plot_visexp(X: pd.DataFrame, y: pd.DataFrame, shap_all: shap.Explanation, feature, trim=None, colorbar=False, y_size=1):
    shap_exp = shap_all.values[:,X.columns.get_loc(feature)]

    X_feature = X[feature]

    if trim == None:
        trim = (0, max(X_feature))

    label = sorted(pd.unique(y))

    y_squish = max(len(y == label[0]), len(y == label[1]))

    # vvv Masking outliers vvv

    mask = (X_feature >= trim[0]) & (X_feature <= trim[1])
    positive_outliers = shap_exp[~mask & (y == label[1])]
    negative_outliers = shap_exp[~mask & (y == label[0])]

    shap_exp = shap_exp[mask]
    X_feature = X_feature[mask]
    y = y[mask]

    # ^^^ Masking outlier ^^^
    
    # The trick is to change size of the plot to make it look nice, it is a bit hacky
    fig, ax = plt.subplots(dpi=100, figsize=(80, y_squish))
    
    cmap_name = "icefire"
    violin_color = "lightgray"

    scaler = max(abs(shap_exp.min()), abs(shap_exp.max()))
    shap_hues = shap_exp / scaler
    shap_hues = (shap_hues + 1) * shap_all.base_values
    
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    # norm = plt.Normalize(vmin=0, vmax=1)
    palette = {h: cmap(h) for h in shap_hues}

    # X_feature = (X_feature - X_feature.min()) / (X_feature.max() - X_feature.min())

    sns.swarmplot(x=X_feature, y=y, order=[label[0], label[1]],
        hue=shap_hues, orient="h", palette=palette,
        size=5)
    

    # Get offsets for negative and positive points
    offsets_negative = np.array(ax.collections[0].get_offsets())
    offsets_positive = np.array(ax.collections[1].get_offsets())
    max_negative = np.max(np.abs(offsets_negative)[:, 1])
    max_positive = np.max(np.abs(offsets_positive)[:, 1] - 1)
    offset_scaler = max(max_negative, max_positive) / 0.35
    
    # Change offset on dots for negative (0) and positive (1)
    offsets_negative = [[elem[0], -abs(elem[1] - 0) / offset_scaler - 0.05] for elem in offsets_negative]
    offsets_positive = [[elem[0], abs(elem[1] - 1) / offset_scaler + 0.05] for elem in offsets_positive]

    ax.collections[0].set_offsets(offsets_negative)
    ax.collections[1].set_offsets(offsets_positive)


    fig.set_size_inches(10, y_size)

    sns.violinplot(x=X_feature, y=[0]*y.size, hue=y, split=True, hue_order=[label[0], label[1]],
        orient="h",  showfliers=False, scale="count", bw=0.2, gridsize=1000, linewidth=0, color=violin_color,
        cut=0, inner=None)
    
    for violin in ax.findobj(matplotlib.collections.PolyCollection):
        violin.set_facecolor("lightgray")
    
    ax.legend_.remove()
    
    

    if colorbar:
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_name), location="bottom", shrink=0.4, anchor=(0.10, 0), pad=0.4)
        cbar.set_label("contribution", labelpad=-55)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels([f"towards\n{y[0]}", "none", f"towards\n{y[1]}"])

        # plt.arrow(0, 0.5, 1, 0, facecolor="black",
        #     width=0.07, head_length=0.7, head_width=0.2,
        #     length_includes_head=True)
    
    
    ticks = np.linspace(trim[0], trim[1], 11)

    # vvv ARROWS vvv

    if positive_outliers.size != 0:
        positive_outliers /= max(abs(positive_outliers.min()), abs(positive_outliers.max()))
        positive_outliers = (positive_outliers + 1) * shap_all.base_values

    if negative_outliers.size != 0:
        negative_outliers /= max(abs(negative_outliers.min()), abs(negative_outliers.max()))
        negative_outliers = (negative_outliers + 1) * shap_all.base_values
    
    largest_outliers = max(len(negative_outliers), len(positive_outliers))

    # If there are "normal" outliers, create arrows to represent them
    if negative_outliers.size != 0:
        # ax.text(last + (last-s_last) * 0.5, -0.28, f"{len(negative_outliers)}")
        arrow_length = 0.5 * ticks[1] * (len(negative_outliers) / largest_outliers)
        ax.arrow(ticks[-1]+ticks[1]*0.3, -0.25, arrow_length, 0,
            facecolor=cmap(negative_outliers.mean()), linewidth=0,
            width=0.09, head_length=(arrow_length)*0.7, head_width=0.3,
            length_includes_head=True)
    # If there are "attack" outliers, create arrows to represent them
    if positive_outliers.size != 0:
        # ax.text(last + (last-s_last) * 0.5, 0.22, f"{len(positive_outliers)}")
        arrow_length = 0.5 * ticks[1] * (len(positive_outliers) / largest_outliers)
        ax.arrow(ticks[-1]+ticks[1]*0.3, 0.25, arrow_length, 0,
            facecolor=cmap(positive_outliers.mean()), linewidth=0,
            width=0.09, head_length=(arrow_length)*0.7, head_width=0.3,
            length_includes_head=True)
    
    # ^^^ ARROWS ^^^

    ax.set_xlabel("")
    ax.set_ylabel(feature) #, rotation="vertical", x=-1, y=0.4)
    ax.set_xticks(ticks=ticks)
    ax.set_yticks(ticks=[-0.25, 0.25], labels=[label[0], label[1]])
    if negative_outliers.size != 0 or positive_outliers.size != 0:
        ax.set_xlim((trim[0]-ticks[1]*0.3, trim[1]+ticks[1]))
    else:
        ax.set_xlim((trim[0]-ticks[1]*0.3, trim[1]+ticks[1]*0.3))

    ax.add_artist(Rectangle((trim[0]-ticks[1]*0.2, -0.01), width=trim[1]+ticks[1]*0.4, height=0.02/y_size, color="black", linewidth=0))
    # ax.axhline(y=0, color="black", linewidth=0.5)
    ax.margins(x=0.5)
    
    # plt.savefig(f"/images/plot_{feature}.pdf", dpi=300, bbox_inches="tight")
    plt.show()