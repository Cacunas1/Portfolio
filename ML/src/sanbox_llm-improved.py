# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
import os

# %%
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# %%
# Load data
data_fp: str = os.path.join("..", "data", "AmesHousing.csv")
df: pl.DataFrame = pl.read_csv(data_fp)

# %%
# Separate numeric and categorical columns
numeric_features = df.select(pl.col(pl.Int64) | pl.col(pl.Float64)).columns
categorical_features = df.select(pl.col(pl.String)).columns

# %%
# Remove target and ID columns
features_to_exclude = ["Order", "PID", "SalePrice"]
numeric_features = [col for col in numeric_features if col not in features_to_exclude]
categorical_features = [
    col for col in categorical_features if col not in features_to_exclude
]

# %%
# Create preprocessing pipelines
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# %%
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        (
            "onehot",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
        ),
    ]
)


# %%
# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# %%
# Create modeling pipeline with Ridge regression
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
    ]
)

# %%
# Prepare data
X = df.drop(["Order", "PID", "SalePrice"])
y = df["SalePrice"]

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=69
)

# %%
# Cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=69)
cv_scores = cross_val_score(
    model, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error"
)

# %%
# Print cross-validation results
mean_cv_score = -cv_scores.mean()
print(f"Mean CV MSE: {mean_cv_score:,.2f}")

# %%
# Fit model
model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model.predict(X_test)

# %%
# Calculate metrics
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
r2 = r2_score(y_test, y_pred)

# %%
print(f"Test MSE: {test_mse:,.2f}")
print(f"Test RMSE: {test_rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# %%
# Get feature names after preprocessing
feature_names = numeric_features + [
    f"{feature}_{val}"
    for feature, vals in zip(
        categorical_features,
        model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .categories_,
    )
    for val in vals[1:]
]  # Skip first category due to drop='first'

# %%
# Create feature importance DataFrame
feature_importance = pl.DataFrame(
    {"feature": feature_names, "importance": abs(model.named_steps["regressor"].coef_)}
)

# %%
# Sort and display top features
feature_importance = feature_importance.sort("importance", descending=True)
print("\nTop 20 Most Important Features:")
with pl.Config(
    tbl_cell_numeric_alignment="RIGHT",
    thousands_separator=",",
    decimal_separator=".",
    float_precision=3,
    tbl_rows=-1,
):
    print(feature_importance.head(20))

# %%
# Additional analysis: Separate numeric and categorical feature importance
# numeric_importance = feature_importance[
#     feature_importance.select(pl.col("feature").is_in(numeric_features))
# ]
# categorical_importance = feature_importance[
#     ~feature_importance.select(pl.col("feature").is_in(numeric_features))
# ]
numeric_importance: pl.DataFrame = feature_importance.filter(
    pl.col("feature").is_in(numeric_features)
)
categorical_importance: pl.DataFrame = feature_importance.filter(
    ~pl.col("feature").is_in(numeric_features)
)


# %%
print("\nTop 10 Most Important Numeric Features:")
with pl.Config(
    tbl_cell_numeric_alignment="RIGHT",
    thousands_separator=",",
    decimal_separator=".",
    float_precision=3,
    tbl_rows=-1,
):
    print(numeric_importance.head(10))

# %%
print("\nTop 10 Most Important Categorical Features:")
with pl.Config(
    tbl_cell_numeric_alignment="RIGHT",
    thousands_separator=",",
    decimal_separator=".",
    float_precision=3,
    tbl_rows=-1,
):
    print(categorical_importance.head(10))
