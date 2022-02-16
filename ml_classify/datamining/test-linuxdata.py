import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
import shap


data: pd.DataFrame = pd.read_csv("mar10_dataset")
X = data.drop(columns="has_ref")
y = data["has_ref"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(f"Train: {np.bincount(y_train)} Test: {np.bincount(y_test)}")

rf = RandomForestClassifier(n_estimators=20, max_depth=20, min_samples_split=3, max_leaf_nodes=100, random_state=0).fit(X, y)

print("Random Forest model fitted!")
avg_depth = 0
avg_leaves = 0
for clf_est in rf.estimators_:
    depth = clf_est.get_depth()
    leaves = clf_est.get_n_leaves()
    print(f"{depth}       {leaves}")
    avg_depth += depth
    avg_leaves += leaves
avg_depth /= len(rf.estimators_)
avg_leaves /= len(rf.estimators_)
print(f"Average depth of trees: {avg_depth}     Average # of leaves: {avg_leaves}")

# dump(clf, "RF_ROAD.joblib")
# print("Model saved!")

# clf = load("RF_Survival.joblib")


scores = cross_val_score(rf, X_train, y_train, scoring='f1', cv=10, n_jobs=-1)
print("Training F1: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

pred = rf.predict(X_test)
print("Test data has been Classified!")

f1_scores = f1_score(y_test, pred, average='weighted')
print("Testing F1:  %0.4f(+/- %0.4f)" % (f1_scores.mean(), f1_scores.std()))


# shap.initjs()
explainer = shap.TreeExplainer(rf)
print("Explainer created!")
shap_values = explainer.shap_values(X_train)
print("Shap values created!")

# shap_values = load("RF_ROAD_Shap.joblib")

# # Make sure that the ingested SHAP model (a TreeEnsemble object) makes the
# # same predictions as the original model
# assert np.abs(explainer.model.predict(X_test) - rf.predict(X_test)).max() < 1e-4

# # make sure the SHAP values sum up to the model output (this is the local accuracy property)
# assert np.abs(explainer.expected_value + explainer.shap_values(X_test).sum(1) - rf.predict(X_test)).max() < 1e-4

shap.summary_plot(shap_values[1], X_train)