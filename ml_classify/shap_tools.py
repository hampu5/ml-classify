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

def plot_exp(df_exp: pd.DataFrame, shap_all: shap.Explanation, feature, trim=None, y_squish=10, scale=False):
    # df_exp = pd.concat([X_train_exp, y_train_exp], axis=1)
    df_exp["Label"].replace({0: "normal", 1: "attack"}, inplace=True)
    shap_exp = shap_all.values[:,df_exp.columns.get_loc(feature)]
    
    feature_max = df_exp[feature].max()

    mask = None
    if trim == None:
        mask = (np.abs(stats.zscore(df_exp[feature])) < 3)
    else:
        feature_max = trim[1]
        mask = (df_exp[feature] > trim[0]) & (df_exp[feature] < trim[1])

    shap_exp = shap_exp[mask]
    df_exp = df_exp[mask]

    # plt.figure(figsize=(20, 2), dpi=100)
    # palette = sns.color_palette("plasma", n_colors=600) #sns.light_palette("seagreen", reverse=False,  n_colors=600 )

    fig, ax = plt.subplots(figsize=(80, y_squish))
    
    cmap_name = "icefire"
    violin_color = "lightgray"

    scaler = max(abs(shap_exp.min()), abs(shap_exp.max()))
    shap_hues = shap_exp / scaler
    shap_hues = (shap_hues + 1) * shap_all.base_values[0]
    
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    norm = plt.Normalize(vmin=0, vmax=1)
    palette = {h: cmap(norm(h)) for h in shap_hues}

    ax = sns.swarmplot(data=df_exp, x=feature, y="Label",
        hue=shap_hues, orient="h", palette=palette,
        size=5)
    ax.legend_.remove()
    
    fig.set_size_inches(20, 2)

    sns.violinplot(data=df_exp, x=feature, y="Label",
        orient="h",  showfliers=False, scale="count", bw=0.3, gridsize=1000, linewidth=0, color=violin_color,
        cut=0, inner=None)
    
    plt.legend([],[], frameon=False)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_name), label="contribution\nof data point", location="right", pad=0.01)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["towards\nnormal", "none", "towards\nattack"])

    plt.title(f"How the RF classifies a sample (600) of training data, viewed through the feature: {feature}")
    feature = feature + " (ms)" if feature[0:2] == "dt" else feature

    plt.ylabel("as classified by RF")
    plt.xlabel(f"value of {feature}")

    plt.xticks(np.append(np.arange(0, feature_max, feature_max / 20), feature_max))
    if scale:
        plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(x*1000, '.2f')))
    else:
        plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(x, '.2f')))
    
    plt.show()