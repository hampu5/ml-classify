import shap
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

shap.initjs()

def get_explanation(explainer, df: pd.DataFrame):

    return_explanation = None

    if isinstance(explainer, shap.TreeExplainer):
        return_explanation = shap.Explanation(explainer(df)[:, :, 1], feature_names=df.columns)
    elif isinstance(explainer, shap.KernelExplainer):
        return_explanation = shap.Explanation(
            values=explainer.shap_values(df)[0],
            base_values=explainer.expected_value,
            data=df.to_numpy(),
            feature_names=df.columns)
    
    # return_explanation = shap.Explanation(
    #     values=explainer.shap_values(df)[0],
    #     base_values=explainer.expected_value,
    #     data=df.to_numpy(),
    #     feature_names=df.columns)

    assert return_explanation != None
    
    return return_explanation

def plot_beeswarm(exp_obj):
    vis = shap.plots.beeswarm(exp_obj, show=False, max_display=20 , color=plt.get_cmap("plasma"))
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    return vis

def plot_waterfall(exp_obj, index):
    vis = shap.plots.waterfall(exp_obj[index])
    return vis

def plot_force(exp_obj):
    vis = shap.plots.force(exp_obj, plot_cmap="DrDb")
    return vis

def plot_dependence(exp_obj, feature, interaction="auto", xmin=None, xmax=None):
    vis = shap.dependence_plot(feature, exp_obj.values,
        pd.DataFrame(exp_obj.data, columns=exp_obj.feature_names),
        interaction_index=interaction,
        xmin=xmin, xmax=xmax,
        alpha=1.0)
    return vis

def plot_heatmap(exp_obj, sort="mean"):
    sorter = exp_obj.abs.mean(0)
    if sort == "max": sorter = exp_obj.abs.max(0)
    vis = shap.plots.heatmap(exp_obj,
        max_display=20,
        show=False,
        feature_values=sorter)
    plt.gcf().axes[-1].set_aspect(200)
    plt.gcf().axes[-1].set_box_aspect(50)
    return vis

def plot_scatter(exp_obj, feature):
    vis = shap.plots.scatter(exp_obj[:,feature], color=exp_obj, show=False)
    plt.show()
    return vis

def plot_exp(df_exp: pd.DataFrame, shap_all: shap.Explanation, feature, trim=None, y_squish=10, scale=1):
    df_exp["Label"].replace({0: "normal", 1: "attack"}, inplace=True)
    shap_exp = shap_all.values[:,df_exp.columns.get_loc(feature)]

    mask = None
    if trim == None:
        mask = (np.abs(stats.zscore(df_exp[feature])) < 3)
    else:
        mask = (df_exp[feature] > trim[0]) & (df_exp[feature] < trim[1])

    attack_outliers = shap_exp[~mask & (df_exp["Label"] == "attack")]
    normal_outliers = shap_exp[~mask & (df_exp["Label"] == "normal")]

    shap_exp = shap_exp[mask]
    df_exp = df_exp[mask]

    # plt.figure(figsize=(20, 2), dpi=100)

    fig, ax = plt.subplots(figsize=(80, y_squish))
    
    cmap_name = "icefire"
    violin_color = "lightgray"

    scaler = max(abs(shap_exp.min()), abs(shap_exp.max()))
    shap_hues = shap_exp / scaler
    shap_hues = (shap_hues + 1) * shap_all.base_values[0]

    if attack_outliers.size != 0:
        attack_outliers /= max(abs(attack_outliers.min()), abs(attack_outliers.max()))
        attack_outliers = (attack_outliers + 1) * shap_all.base_values[0]

    if normal_outliers.size != 0:
        normal_outliers /= max(abs(normal_outliers.min()), abs(normal_outliers.max()))
        normal_outliers = (normal_outliers + 1) * shap_all.base_values[0]
    
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    norm = plt.Normalize(vmin=0, vmax=1)
    palette = {h: cmap(h) for h in shap_hues}

    values = df_exp[feature]
    feature_min = values.min()
    feature_max = values.max()
    values = (values - values.min()) / (values.max() - values.min())
    label = df_exp["Label"]

    ax_swarm = sns.swarmplot(x=values, y=label, order=["attack", "normal"],
        hue=shap_hues, orient="h", palette=palette,
        size=5)
    ax.legend_.remove()

    fig.set_size_inches(20, 3)

    sns.violinplot(x=values, y=label, order=["attack", "normal"],
        orient="h",  showfliers=False, scale="count", bw=0.3, gridsize=1000, linewidth=0, color=violin_color,
        cut=0, inner=None, ax=ax_swarm)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_name), label="contribution", location="bottom", shrink=0.2, anchor=(1, 0.9))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["towards\nnormal", "none", "towards\nattack"])

    plt.title(f"How the RF-model classifies data - feature: {feature}")
    feature = feature + " (ms)" if feature[0:2] == "dt" else feature

    plt.ylabel("type of data")
    plt.xlabel(f"value of {feature}")

    plt.xticks(ticks=np.linspace(0, 1, 20), labels=map(lambda x: format(x*scale, '.2f'), np.linspace(feature_min, feature_max, 20)))
    
    s_last = ax.get_xticks()[-2]
    last = ax.get_xticks()[-1]
    
    if attack_outliers.size != 0:
        plt.arrow(last + (last-s_last) * 0.5, 0, last-s_last, 0, facecolor=cmap(attack_outliers.mean()), edgecolor=violin_color,
            width=0.15, head_length=(last-s_last)*0.7, head_width=0.4,
            length_includes_head=True)
    if normal_outliers.size != 0:
        plt.arrow(last + (last-s_last) * 0.5, 1, last-s_last, 0, facecolor=cmap(normal_outliers.mean()), edgecolor=violin_color,
            width=0.15, head_length=(last-s_last)*0.7, head_width=0.4,
            length_includes_head=True)
    
    plt.margins(x=0.02)
    
    plt.show()