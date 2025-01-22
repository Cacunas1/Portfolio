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

# %% [markdown]
# # Linear Regression Showcase
#
# The idea is to showcase how a linear regression model is created, what considerations have to be taken into consideration, and how the result can be used, an interpreted

# %% [markdown]
# ## Imports

# %%
import os
import polars as pl
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error

# %% [markdown]
# ## Data Import

# %%
data_fp: str = os.path.join("..", "data", "AmesHousing.csv")

df: pl.DataFrame = pl.read_csv(data_fp)

# %% [markdown]
# ## Exploratory Data Analysis

# %%
df.schema

# %%
df.columns

# %%
df.glimpse()

# %%
df.describe()

# %%
df.head()

# %%
rcol = "House Style"
ht_cats = df.get_column(rcol).unique().to_list()
print(ht_cats)

for cat in ht_cats:
    aux = df.filter(pl.col(rcol) == pl.lit(cat))
    fig, ax = plt.subplots()
    ax.plot(
        aux.sort("Lot Area").select("Lot Area"),
        aux.sort("Lot Area").select("SalePrice"),
    )
    ax.grid()
    plt.show()

# %% [markdown]
# ## Transform (preprocessing)

# %%
df = df.with_columns(pl.col(pl.Int64).fill_nan(0).fill_null(0))

# %%
df = df.with_columns(pl.col(pl.String).fill_null("NULL"))

# %%
df.select(pl.col(pl.String)).glimpse()


# %%
def category_to_int(cat: pl.Categorical) -> pl.Int8:
    match cat:
        case "A (agr)":
            return 0
        case "RL":
            return 1
        case "RM":
            return 2
        case "RH":
            return 3
        case "C (all)":
            return 4
        case "FV":
            return 5
        case "I (all)":
            return 6
        case _:
            return -1


# %%
df = df.with_columns(
    pl.col("MS Zoning")
    .map_elements(category_to_int, return_dtype=pl.Int64)
    .cast(pl.Int64)
)

# %%
df = df.with_columns(pl.col(pl.String).cast(pl.Categorical))

# %%
df = df.to_dummies(pl.selectors.categorical())

# %%
df.schema

# %% [markdown]
# ## Train/Test separation

# %%
FEATURES: list[str] = list(set(df.columns).difference({"Order", "PID", "SalePrice"}))

TARGET: str = "SalePrice"

X: pl.DataFrame = df.select(FEATURES)
y: pl.DataFrame = df.select(TARGET)

X_train: pl.DataFrame
X_test: pl.DataFrame
y_train: pl.DataFrame
y_test: pl.DataFrame

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=69
)

# %% [markdown]
# ## Cross-validation and training

# %%
lr_model: skl.linear_model = LinearRegression()

# %%
kfold: KFold = KFold(n_splits=10, shuffle=True, random_state=69)

# %%
# testing
# X_train = X_train.select(pl.col(pl.Int64))

# %%
cv_scores = cross_val_score(
    lr_model, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error"
)

# %%
cv_scores

# %%
# Calculate the mean of the cross-validation scores
mean_cv_score = -cv_scores.mean()
print(f"Mean CV MSE: {mean_cv_score:_}")

# %%
# Fit the model on the full training set
lr_model.fit(X_train, y_train)

# %%
# Predict on the test set
y_pred = lr_model.predict(X_test)

# %%
# Calculate the Mean Squared Error on the test set
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {test_mse:_}")

# %%
y_pred[:10]

# %%
y_pred = y_pred.flatten()
y_pred = pl.DataFrame({"SalesPrice": y_pred})

# %%
y_pred = y_pred.select(pl.col("SalesPrice").round().cast(pl.Int64))

# %%
y_pred

# %%
y_test

# %%
y_err = y_pred.select(pl.all()) - y_test.select(pl.all())
