import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load users data
users = pd.read_csv("Users.csv")
print(users.head())

# Load books data
books = pd.read_csv("Books.csv")
print(books.head())

# Load ratings data
ratings = pd.read_csv("Ratings.csv")
print(ratings.head())


def missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = round(df.isnull().mean().mul(100), 2)
    mz_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(columns={df.index.name: 'col_name', 0: 'Missing Values', 1: '% of Total Values'})
    mz_table['Data_type'] = df.dtypes
    mz_table = mz_table.sort_values('% of Total Values', ascending=False)
    return mz_table.reset_index()
