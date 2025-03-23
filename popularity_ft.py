import pandas as pd
import seaborn as sns

# Load datasets
users = pd.read_csv("Users.csv")
books = pd.read_csv("Books.csv", dtype={'ISBN': str}, low_memory=False)
ratings = pd.read_csv("Ratings.csv")

# Filter ratings to only include books that exist in books.csv
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]

# Separate explicit ratings
ratings_explicit = ratings_new[ratings_new['Book-Rating'] != 0].copy()

# Compute mean rating and total number of users rated
ratings_explicit.loc[:, 'Avg_Rating'] = ratings_explicit.groupby('ISBN')['Book-Rating'].transform('mean')
ratings_explicit.loc[:, 'Total_No_Of_Users_Rated'] = ratings_explicit.groupby('ISBN')['Book-Rating'].transform('count')

# Merge with books dataset
Final_Dataset = pd.merge(ratings_explicit, books, on='ISBN')

# Calculate C (mean rating) and m (90th percentile of total ratings)
C = Final_Dataset['Avg_Rating'].mean()
m = Final_Dataset['Total_No_Of_Users_Rated'].quantile(0.90)

# Select books with at least m ratings
Top_Books = Final_Dataset.loc[Final_Dataset['Total_No_Of_Users_Rated'] >= m].copy()


# Define the weighted rating function
def weighted_rating(x, m=m, C=C):
    v = x['Total_No_Of_Users_Rated']
    R = x['Avg_Rating']
    return (v / (v + m) * R) + (m / (m + v) * C)


# Apply weighted rating
Top_Books.loc[:, 'Score'] = Top_Books.apply(weighted_rating, axis=1)

# Sort books by score
Top_Books = Top_Books.drop_duplicates('ISBN').sort_values('Score', ascending=False)

# Print the top 20 books
top_20_books = Top_Books[['Book-Title', 'Total_No_Of_Users_Rated', 'Avg_Rating', 'Score']].reset_index(drop=True).head(
    20)
print(top_20_books.to_string())
