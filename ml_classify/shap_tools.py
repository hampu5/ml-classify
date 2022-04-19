import shap
import pandas as pd
import matplotlib.pyplot as plt

shap.initjs()

def get_explanation(explainer, df: pd.DataFrame, size):
    size = min(len(df), size)
    df = df.sample(size, random_state=0)

    if isinstance(explainer, shap.TreeExplainer):
        return shap.Explanation(explainer(df)[:, :, 1], feature_names=df.columns)
    elif isinstance(explainer, shap.KernelExplainer):
        return shap.Explanation(
            values=explainer.shap_values(df)[0],
            base_values=explainer.expected_value,
            data=df.to_numpy(),
            feature_names=df.columns)
    return None

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

def plot_scatter(exp_obj, feature):
    vis = shap.plots.scatter(exp_obj[:,feature], color=exp_obj)
    return vis