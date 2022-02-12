import shap
from joblib import load

shap_values = load("RF_ROAD_Shap.joblib")
print(shap_values)
# tmp = shap.Explanation(shap_values[:, :, 1])


# shap.plots.beeswarm(tmp)