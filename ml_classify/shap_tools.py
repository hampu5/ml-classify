import shap
import pandas as pd
import matplotlib.pyplot as plt

shap.initjs()

def get_explanation(explainer, df: pd.DataFrame):
    size = min(len(df), 600)
    df = df.sample(size, random_state=0)

    shap_value = explainer(df)
    shap_value = shap.Explanation(shap_value[:, :, 1], feature_names=df.columns)
    return shap_value

def plot_beeswarm(exp_obj):
    vis = shap.plots.beeswarm(exp_obj, show=False, color=plt.get_cmap("plasma"))
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    return vis

def plot_waterfall(exp_obj):
    vis = shap.plots.waterfall(exp_obj[0])
    return vis