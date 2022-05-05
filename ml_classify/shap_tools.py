import shap
import pandas as pd
import matplotlib.pyplot as plt

shap.initjs()

def get_explanation(explainer, df: pd.DataFrame, size):
    size = min(len(df), size)
    df = df.sample(size, random_state=0)

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