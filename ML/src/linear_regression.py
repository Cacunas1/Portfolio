#!/usr/bin/env python3
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
# %%
import os

# %%
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import polars as pl
import numpy as np

# %%
import matplotlib.pyplot as plt

# %%
from sklearn.metrics import accuracy_score, roc_auc_score


# %%
def main():
    data_fp: str = os.path.join("..", "data", "AmesHousing.csv")

    df: pl.DataFrame = pl.read_csv(data_fp)

    print(f"df:\n{df.head()}")

    for col in df.columns:
        print(f"\t- {col}")

    # X: pl.DataFrame = df.select(pl.exclude("SalePrice"))
    # y: pl.DataFrame = df.select

    # split the data for testing and cross-validation
    FEATURES: list[str] = [
        "MS SubClass",
        "MS Zoning",
        "Lot Frontage",
        "Lot Area",
        "Street",
        "Alley",
        "Lot Shape",
        "Land Contour",
        "Utilities",
        "Lot Config",
        "Land Slope",
        "Neighborhood",
        "Condition 1",
        "Condition 2",
        "Bldg Type",
        "House Style",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Year Remod/Add",
        "Roof Style",
        "Roof Matl",
        "Exterior 1st",
        "Exterior 2nd",
        "Mas Vnr Type",
        "Mas Vnr Area",
        "Exter Qual",
        "Exter Cond",
        "Foundation",
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "RsmtFin Type 1",
        "BsmtFin SF 1",
        "BsmtFin Type 2",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "Heating",
        "Heating QC",
        "Central Air",
        "Electrical",
        "1st Flr SF",
        "2nd Flr SF",
        "Low Qual Fin SF",
        "Gr Liv Area",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
        "Full Bath",
        "Half Bath",
        "Bedroom AbvGr",
        "Kitchen AbvGr",
        "Kitchen Qual",
        "TotRms AbvGrd",
        "Functional",
        "Fireplaces",
        "Fireplace Qu",
        "Garage Type",
        "Garage Yr Blt",
        "Garage Finish",
        "Garage Cars",
        "Garage Area",
        "Garage Qual",
        "Garage Cond",
        "Paved Drive",
        "Wood Deck SF",
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch",
        "Pool Area",
        "Pool QC",
        "Fence",
        "Misc Feature",
        "Misc Val",
        "Mo Sold",
        "Yr Sold",
        "Sale Type",
        "Sale Condition",
    ]

    TARGET: str = "SalePrice"

    X: pl.DataFrame = df.select(FEATURES)
    y: pl.DataFrame = df.select(TARGET)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)


# %%
type(X_train)

# %%
if __name__ == "__main__":
    main()
