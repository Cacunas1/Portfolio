#!/usr/bin/env python3

import os
# import keras
import polars as pl

def main():
    data_fp: str = os.path.join("..", "data", "AmesHousing.csv")

    df: pl.DataFrame = pl.read_csv(data_fp)

    print(f"df:\n{df.head()}")

    for col in df.columns:
        print(f"\t- {col}")


if __name__ == "__main__":
    main()
