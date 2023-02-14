import pandas as pd
import os


def get_data(nrows=10000):
    '''returns a DataFrame with nrows from s3 bucket'''
    csv_file = "data/train.csv"
    df = pd.read_csv(csv_file, nrows=nrows)
    return df


def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df.query(
        "fare_amount > 0 & passenger_count <= 8 & passenger_count > 0")
    df = df[df["pickup_latitude"].between(left = 40, right = 42 )]
    df = df[df["pickup_longitude"].between(left = -74.3, right = -72.9 )]
    df = df[df["dropoff_latitude"].between(left = 40, right = 42 )]
    df = df[df["dropoff_longitude"].between(left = -74, right = -72.9 )]
    return df
