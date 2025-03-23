# Importing modules
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Load datasets
users = pd.read_csv("Users.csv")
books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")

def missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = round(df.isnull().mean().mul(100), 2)
    mz_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mz_table.columns = ['Missing Values', '% of Total Values']
    return mz_table

# Print missing values summary
print(missing_values(books))
print(missing_values(users))
print(missing_values(ratings))

# Process users dataset
print(sorted(users.Age.unique()))  # Print sorted unique ages

# Extract country from location
users['Country'] = users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')
print(users.Country.nunique())  # Count unique country names

# Drop the Location column
users.drop('Location', axis=1, inplace=True)

# Clean and standardize country names
users['Country'] = users['Country'].astype('str')
unique_countries = list(set(users.Country.unique()))
unique_countries = [x for x in unique_countries if x is not None]
unique_countries.sort()
print(unique_countries)

users['Country'].replace(
    ['', '01776', '02458', '19104', '23232', '30064', '85021', '87510', 'alachua', 'america',
     'austria', 'autralia', 'cananda', 'geermany', 'italia', 'united kindgonm', 'united sates',
     'united staes', 'united state', 'united states', 'us'],
    ['other', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa',
     'australia', 'australia', 'canada', 'germany', 'italy', 'united kingdom',
     'usa', 'usa', 'usa', 'usa', 'usa'],
    inplace=True)

# Handle outliers in Age
users.loc[(users.Age > 100) | (users.Age < 5), 'Age'] = np.nan

# Fill missing Age values
print(users.isna().sum())  # Count missing values

users['Age'] = users['Age'].fillna(users.groupby('Country')['Age'].transform('median'))
users['Age'].fillna(users.Age.mean(), inplace=True)
