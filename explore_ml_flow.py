import mlflow
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
import matplotlib.pyplot as plt

from belle_bot.mapping.positioning.config.positioning_config import MlFlowConfig

if __name__ == "__main__":
    mlflow.set_tracking_uri(MlFlowConfig().endpoint)

    # Download runs as a pandas table
    runs_df = mlflow.search_runs(
        experiment_names=["positioning"],
        # filter_string='tags.experiment = "all 5"'
        filter_string='tags.experiment = "all 10"'
    )

    # 3. Clean and prepare the data for the decision tree
    # MLflow prepends 'params.' to parameters and 'metrics.' to results/metrics
    print("Available columns:", runs_df.columns)

    # Select the parameters you want to use as features (X)
    # Example: if you logged 'lr' and 'batch_size' as params
    feature_cols = [col for col in runs_df.columns if col.startswith("params.")]
    X = runs_df[feature_cols].copy()
    X = X.replace({
        'True': 1, 'true': 1, 'TRUE': 1,
        'False': 0, 'false': 0, 'FALSE': 0,
        'None': None, 'none': None
    })

    # Convert all columns to numeric, non-numeric become NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Select the metric you want to use as the target result (y)
    # Example: fitting a tree to understand what drives your 'accuracy' or 'loss'
    target_metric = "metrics.mean_step_error_window"
    # target_metric = "metrics.mean_position_error"
    # target_metric = "metrics.mean_final_position_error"
    y = runs_df[target_metric]

    # Fill missing values if any runs failed or skipped logging certain parameters
    X = X.fillna(0)
    y = y.dropna()  # Remove runs that don't have the target metric
    X = X.loc[y.index]  # Align features with the remaining targets

    print(len(X))

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X, np.log1p(y))

    best_y_indices = y < sorted(y)[min(21, len(y) - 1)]
    worst_y_indices = y > sorted(y)[min(21, len(y) - 1)]

    # Normalize X to [0, 1] for comparison
    X_min = X.min()
    X_max = X.max()
    X_norm = (X - X_min) / (X_max - X_min)
    X_norm = X_norm.fillna(0)  # In case max == min

    best_norm_means = X_norm[best_y_indices].mean()
    worst_norm_means = X_norm[worst_y_indices].mean()

    discrepancies = (best_norm_means - worst_norm_means).abs()

    print("\nDiscrepancies in normalized columns (Best vs Worst):")
    print(discrepancies.sort_values(ascending=False))

    highest_discrepancy_col = discrepancies.idxmax()
    highest_val = discrepancies.max()
    print(f"\nHighest discrepancy column: {highest_discrepancy_col} with value {highest_val:.4f}")

    # print("Decision Tree successfully fitted to MLflow run results!")
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    # tree.plot_tree(rf_model.estimators_[0], feature_names=X.columns, filled=True, class_names=True)
    # plt.show()

    correlations = X.apply(lambda col: pd.Series(col).corr(y, method='spearman'))
    print("Spearman Correlation with Error Metric:")
    print(correlations.sort_values())

    "Mean"
    print(np.mean(y))